import UIKit
import WebKit

/// The page's requests to the app: window.webkit.messageHandlers.aiet.postMessage({ action, … }) in app.js, each
/// answered through the promise postMessage returns. Only the site's own pages, in the main frame, are heard.
final class BridgeHandler: NSObject, WKScriptMessageHandlerWithReply {
    /// Weak: the web view keeps its handlers, and the controller keeps the web view.
    private weak var owner: WebController?

    init(owner: WebController) {
        self.owner = owner
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage,
                               replyHandler: @escaping (Any?, String?) -> Void) {
        guard message.frameInfo.isMainFrame, AppConfig.isSiteHost(message.frameInfo.securityOrigin.host),
              let body = message.body as? [String: Any], let action = body["action"] as? String else {
            replyHandler(nil, "Only AI Every Time's own pages can ask the app for this.")
            return
        }
        let owner = self.owner
        Task { @MainActor in
            guard let owner else {
                replyHandler(nil, "The app isn't ready yet.")
                return
            }
            do {
                replyHandler(try await owner.handle(action, body), nil)
            } catch {
                replyHandler(nil, error.localizedDescription)
            }
        }
    }
}

extension WebController {
    enum BridgeError: LocalizedError {
        case unknown(String), nothingToShare

        var errorDescription: String? {
            switch self {
            case .unknown(let action): return "This version of the app can't do \"\(action)\". Update it from the App Store."
            case .nothingToShare: return "There's nothing to share."
            }
        }
    }

    /// What the page asked for, and the answer its promise resolves with:
    /// - pushStatus: whether notifications are allowed, and this device's token if it has one;
    /// - enablePush: asks permission and registers with Apple's push service, then gives the token and its environment;
    /// - openSettings: opens the app's page in Settings, where notifications can be turned back on;
    /// - share: the share sheet, with the text and link the page gives.
    func handle(_ action: String, _ body: [String: Any]) async throws -> Any? {
        switch action {
        case "pushStatus":
            return await PushManager.shared.status()
        case "enablePush":
            let token = try await PushManager.shared.enable()
            return ["token": token, "environment": AppConfig.pushEnvironment]
        case "openSettings":
            if let url = URL(string: UIApplication.openSettingsURLString) {
                _ = await UIApplication.shared.open(url)
            }
            return true
        case "share":
            var items: [Any] = []
            if let text = body["text"] as? String, !text.isEmpty { items.append(text) }
            if let link = body["url"] as? String, let url = URL(string: link) { items.append(url) }
            if items.isEmpty { throw BridgeError.nothingToShare }
            share(items)
            return true
        default:
            throw BridgeError.unknown(action)
        }
    }
}
