import Combine
import Network
import SafariServices
import UIKit
import WebKit

/// The web view that shows AI Every Time, and what the app does around it: links that leave the site open outside it,
/// downloads go to the share sheet, the page's alerts show as the system's, the page's requests to the app are
/// answered (Bridge.swift), and a lost connection shows the offline screen until it comes back.
@MainActor
final class WebController: NSObject, ObservableObject {
    static let shared = WebController()

    let webView: WKWebView
    /// How far the current page has loaded, for the progress bar.
    @Published private(set) var progress: Double = 0
    @Published private(set) var isLoading = false
    /// The first page has loaded, or failed to: the launch cover can go.
    @Published private(set) var started = false
    /// Why the site couldn't be reached, while the offline screen shows.
    @Published private(set) var failure: String?

    private var lastURL = AppConfig.startURL
    private var downloads: [ObjectIdentifier: URL] = [:]
    private let network = NWPathMonitor()

    override init() {
        let configuration = WKWebViewConfiguration()
        configuration.applicationNameForUserAgent = AppConfig.userAgentName
        configuration.allowsInlineMediaPlayback = true
        // Phones and iPads get the site's own layouts for their width, as in Safari.
        configuration.defaultWebpagePreferences.preferredContentMode = .mobile
        webView = WKWebView(frame: .zero, configuration: configuration)
        super.init()
        webView.configuration.userContentController.addScriptMessageHandler(
            BridgeHandler(owner: self), contentWorld: .page, name: "aiet")
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsBackForwardNavigationGestures = true
        webView.allowsLinkPreview = false
        // The page reaches the screen's edges and keeps clear of the status bar and home bar itself (styles.css).
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        webView.isOpaque = false
        webView.backgroundColor = UIColor(named: "LaunchBackground")
        #if DEBUG
        if #available(iOS 16.4, *) {
            webView.isInspectable = true  // Safari's Web Inspector, for builds run from Xcode
        }
        #endif
        webView.publisher(for: \.estimatedProgress).receive(on: RunLoop.main).assign(to: &$progress)
        webView.publisher(for: \.isLoading).receive(on: RunLoop.main).assign(to: &$isLoading)
        watchNetwork()
        webView.load(URLRequest(url: AppConfig.startURL))
    }

    /// Opens a page of the site, such as the one a notification points to.
    func open(path: String) {
        let url = AppConfig.page(path)
        guard AppConfig.isSite(url) else { return }
        failure = nil
        webView.load(URLRequest(url: url))
    }

    /// Loads the page again after it failed to load.
    func retry() {
        failure = nil
        webView.load(URLRequest(url: lastURL))
    }

    /// Tries again by itself when the connection comes back while the offline screen shows.
    private func watchNetwork() {
        network.pathUpdateHandler = { [weak self] path in
            guard path.status == .satisfied else { return }
            Task { @MainActor in
                if let self, self.failure != nil { self.retry() }
            }
        }
        network.start(queue: DispatchQueue(label: "org.aieverytime.network"))
    }

    /// An address off the site: web pages open in Safari's viewer over the app; mail, phone and the like in their apps.
    func openOutside(_ url: URL) {
        if ["http", "https"].contains(url.scheme?.lowercased() ?? ""), let top = Presenter.top {
            let safari = SFSafariViewController(url: url)
            safari.preferredControlTintColor = UIColor(named: "AccentColor")
            top.present(safari, animated: true)
        } else {
            UIApplication.shared.open(url)
        }
    }

    /// The share sheet, for a downloaded file or something the page shares.
    func share(_ items: [Any]) {
        guard let top = Presenter.top else { return }
        let sheet = UIActivityViewController(activityItems: items, applicationActivities: nil)
        if let popover = sheet.popoverPresentationController {  // on iPad it's a popover, which needs a place to point
            popover.sourceView = top.view
            popover.sourceRect = CGRect(x: top.view.bounds.midX, y: top.view.bounds.midY, width: 0, height: 0)
            popover.permittedArrowDirections = []
        }
        top.present(sheet, animated: true)
    }

    private func show(_ alert: UIAlertController, otherwise: () -> Void) {
        guard let top = Presenter.top else {
            otherwise()
            return
        }
        top.present(alert, animated: true)
    }

    private func fail(_ error: Error) {
        let error = error as NSError
        // Not failures: a load replaced by another, or one that became a download or went to Safari's viewer.
        if error.domain == NSURLErrorDomain && error.code == NSURLErrorCancelled { return }
        if error.domain == "WebKitErrorDomain" && error.code == 102 { return }
        if let url = error.userInfo[NSURLErrorFailingURLErrorKey] as? URL, AppConfig.isSite(url) { lastURL = url }
        failure = error.localizedDescription
        started = true
    }
}

// MARK: - Links and loading

extension WebController: WKNavigationDelegate {
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if navigationAction.shouldPerformDownload {  // a link with the download attribute
            decisionHandler(.download)
            return
        }
        guard let url = navigationAction.request.url, let scheme = url.scheme?.lowercased() else {
            decisionHandler(.cancel)
            return
        }
        if ["about", "blob", "data"].contains(scheme) || AppConfig.isSite(url) {
            decisionHandler(.allow)
        } else if (scheme == "http" || scheme == "https") && navigationAction.targetFrame?.isMainFrame == false {
            decisionHandler(.allow)  // something a page embeds, such as a video
        } else {
            decisionHandler(.cancel)
            openOutside(url)
        }
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        let disposition = (navigationResponse.response as? HTTPURLResponse)?
            .value(forHTTPHeaderField: "Content-Disposition")?.lowercased() ?? ""
        decisionHandler(disposition.hasPrefix("attachment") || !navigationResponse.canShowMIMEType ? .download : .allow)
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        failure = nil
        started = true
        if let url = webView.url, AppConfig.isSite(url) { lastURL = url }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        fail(error)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        fail(error)
    }

    /// The page's process can be stopped when the phone runs short of memory, which leaves a blank view: load it again.
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        webView.load(URLRequest(url: webView.url ?? lastURL))
    }
}

// MARK: - New windows, the page's alerts, the microphone and camera

extension WebController: WKUIDelegate {
    /// A link that opens a new window: the site's pages open here, any other address outside the app.
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = navigationAction.request.url {
            if AppConfig.isSite(url) {
                webView.load(navigationAction.request)
            } else if url.scheme?.lowercased() != "about" {
                openOutside(url)
            }
        }
        return nil
    }

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let alert = UIAlertController(title: nil, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default) { _ in completionHandler() })
        show(alert, otherwise: completionHandler)
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let alert = UIAlertController(title: nil, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel) { _ in completionHandler(false) })
        alert.addAction(UIAlertAction(title: "OK", style: .default) { _ in completionHandler(true) })
        show(alert) { completionHandler(false) }
    }

    func webView(_ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String, defaultText: String?,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (String?) -> Void) {
        let alert = UIAlertController(title: nil, message: prompt, preferredStyle: .alert)
        alert.addTextField { $0.text = defaultText }
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel) { _ in completionHandler(nil) })
        alert.addAction(UIAlertAction(title: "OK", style: .default) { [weak alert] _ in
            completionHandler(alert?.textFields?.first?.text ?? "")
        })
        show(alert) { completionHandler(nil) }
    }

    /// The site may use the microphone and camera; the phone still asks the learner the first time (Info.plist says why).
    func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
                 initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
                 decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        decisionHandler(AppConfig.isSiteHost(origin.host) ? .grant : .deny)
    }
}

// MARK: - Downloads

extension WebController: WKDownloadDelegate {
    /// A download, such as a skill's zip or a PDF, goes to a folder of its own, then to the share sheet to keep.
    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String,
                  completionHandler: @escaping (URL?) -> Void) {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("Downloads", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        } catch {
            completionHandler(nil)
            return
        }
        let file = folder.appendingPathComponent(suggestedFilename.isEmpty ? "download" : suggestedFilename)
        downloads[ObjectIdentifier(download)] = file
        completionHandler(file)
    }

    func downloadDidFinish(_ download: WKDownload) {
        guard let file = downloads.removeValue(forKey: ObjectIdentifier(download)) else { return }
        share([file])
    }

    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
        downloads.removeValue(forKey: ObjectIdentifier(download))
        let alert = UIAlertController(title: "The download didn't finish", message: error.localizedDescription,
                                      preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        show(alert) {}
    }
}

/// The view controller on top, to show a sheet, an alert or Safari's viewer over.
enum Presenter {
    @MainActor static var top: UIViewController? {
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let window = scenes.flatMap { $0.windows }.first { $0.isKeyWindow } ?? scenes.first?.windows.first
        var top = window?.rootViewController
        while let presented = top?.presentedViewController { top = presented }
        return top
    }
}
