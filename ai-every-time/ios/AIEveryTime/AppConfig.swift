import Foundation
import UIKit

/// Where the app's pages come from, which addresses stay inside the app, and how the app names itself to the site.
enum AppConfig {
    /// The site the app shows. It comes from the AIET_SITE_URL build setting (Info.plist's AIETSiteURL), so a build
    /// can point at a test copy of the site; only https addresses are used.
    static let siteURL: URL = {
        if let text = Bundle.main.object(forInfoDictionaryKey: "AIETSiteURL") as? String,
           let url = URL(string: text), url.scheme == "https", url.host != nil {
            return url
        }
        return URL(string: "https://www.aieverytime.org")!
    }()

    /// The page the app opens on: the feed, which sends someone not signed in to the welcome page.
    static var startURL: URL { page("/#/feed") }

    /// A page of the site from its path, such as "/#/review" from a notification.
    static func page(_ path: String) -> URL {
        URL(string: path, relativeTo: siteURL)?.absoluteURL ?? siteURL
    }

    /// The site's own host, with and without "www.": its pages open in the app, and only they can ask the app for
    /// anything. Any other address opens in Safari's viewer.
    static func isSiteHost(_ host: String?) -> Bool {
        guard let host = host?.lowercased(), let site = siteURL.host?.lowercased() else { return false }
        let root = site.hasPrefix("www.") ? String(site.dropFirst(4)) : site
        return host == root || host == "www." + root
    }

    static func isSite(_ url: URL?) -> Bool {
        url?.scheme?.lowercased() == "https" && isSiteHost(url?.host)
    }

    /// Added to the web view's user agent, so the site knows it's in the app (app.js looks for "AIEveryTimeApp/"),
    /// and which device it's on, for the learner's list of devices.
    @MainActor static var userAgentName: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0"
        let kind = UIDevice.current.userInterfaceIdiom == .pad ? "iPad" : "iPhone"
        return "AIEveryTimeApp/\(version) (\(kind))"
    }

    /// Which of Apple's push services this build's device tokens belong to: "sandbox" for builds run from Xcode,
    /// "production" for TestFlight and the App Store. The provisioning profile inside the app says which; App Store
    /// and TestFlight builds carry none.
    static let pushEnvironment: String = {
        guard let url = Bundle.main.url(forResource: "embedded", withExtension: "mobileprovision"),
              let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .isoLatin1),
              let start = text.range(of: "<?xml"), let end = text.range(of: "</plist>"),
              let plist = try? PropertyListSerialization.propertyList(
                  from: Data(text[start.lowerBound..<end.upperBound].utf8), options: [], format: nil) as? [String: Any],
              let entitlements = plist["Entitlements"] as? [String: Any],
              let environment = entitlements["aps-environment"] as? String
        else { return "production" }
        return environment == "development" ? "sandbox" : "production"
    }()
}
