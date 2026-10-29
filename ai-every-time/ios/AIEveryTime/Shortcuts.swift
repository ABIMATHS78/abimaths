import AppIntents
import UIKit

/// Ways into the app from outside it, each opening a page of the site: the actions on the app's icon (touch and
/// hold it on the Home Screen, listed in Info.plist), and shortcuts for Siri, Spotlight and the Shortcuts app.
enum Shortcut: String {
    case companion, learn, practice, courses

    var path: String {
        switch self {
        case .companion: "/my-companion"  // the learner's own companion, which the site knows and the app doesn't
        case .learn: "/learn"
        case .practice: "/practice"
        case .courses: "/courses"
        }
    }

    /// Opens the page for an action chosen on the app's icon. False for one the app doesn't know.
    @MainActor @discardableResult
    static func open(_ item: UIApplicationShortcutItem) -> Bool {
        guard let shortcut = Shortcut(rawValue: item.type) else { return false }
        WebController.shared.open(path: shortcut.path)
        return true
    }
}

/// Hears about an action chosen on the app's icon, whether it started the app or found it running.
final class SceneDelegate: NSObject, UIWindowSceneDelegate {
    func scene(_ scene: UIScene, willConnectTo session: UISceneSession,
               options connectionOptions: UIScene.ConnectionOptions) {
        if let item = connectionOptions.shortcutItem { Shortcut.open(item) }
    }

    func windowScene(_ windowScene: UIWindowScene, performActionFor shortcutItem: UIApplicationShortcutItem,
                     completionHandler: @escaping (Bool) -> Void) {
        completionHandler(Shortcut.open(shortcutItem))
    }
}

// MARK: - Siri, Spotlight and the Shortcuts app

struct ChatWithCompanion: AppIntent {
    static let title: LocalizedStringResource = "Chat with my companion"
    static let description = IntentDescription("Opens the chat with your companion in AI Every Time.")
    static let openAppWhenRun = true

    @MainActor
    func perform() async throws -> some IntentResult {
        WebController.shared.open(path: Shortcut.companion.path)
        return .result()
    }
}

struct ContinueLearning: AppIntent {
    static let title: LocalizedStringResource = "Continue learning"
    static let description = IntentDescription("Opens your learning paths in AI Every Time, where you left off.")
    static let openAppWhenRun = true

    @MainActor
    func perform() async throws -> some IntentResult {
        WebController.shared.open(path: Shortcut.learn.path)
        return .result()
    }
}

struct PractiseFlashcards: AppIntent {
    static let title: LocalizedStringResource = "Practise flashcards"
    static let description = IntentDescription("Opens your flashcards and quizzes in AI Every Time.")
    static let openAppWhenRun = true

    @MainActor
    func perform() async throws -> some IntentResult {
        WebController.shared.open(path: Shortcut.practice.path)
        return .result()
    }
}

struct AIEveryTimeShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(intent: ChatWithCompanion(), phrases: [
            "Chat with my companion in \(.applicationName)",
            "Open my companion in \(.applicationName)",
        ])
        AppShortcut(intent: ContinueLearning(), phrases: [
            "Continue learning in \(.applicationName)",
            "Learn with \(.applicationName)",
        ])
        AppShortcut(intent: PractiseFlashcards(), phrases: [
            "Practise flashcards in \(.applicationName)",
            "Practice flashcards in \(.applicationName)",
        ])
    }
}
