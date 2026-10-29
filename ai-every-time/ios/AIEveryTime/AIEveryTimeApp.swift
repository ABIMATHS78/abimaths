import SwiftUI

/// AI Every Time for iPhone and iPad: the site's own pages in a web view, with what only an app can add around them
/// (notifications through Apple, the share sheet, downloads, an offline screen). See ios/README.md.
@main
struct AIEveryTimeApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView(web: WebController.shared)
        }
    }
}
