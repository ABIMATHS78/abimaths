import UIKit
import UserNotifications

/// Notifications through Apple's push service. The page asks for them (Bridge.swift); the app asks the learner's
/// permission, registers the device with Apple, and gives the page the device's token, which it passes to the server.
@MainActor
final class PushManager {
    static let shared = PushManager()

    enum Failure: LocalizedError {
        case denied, timedOut

        var errorDescription: String? {
            switch self {
            case .denied: return "Notifications are off for AI Every Time. You can turn them on in Settings."
            case .timedOut: return "Apple's notification service didn't answer. Check your connection and try again."
            }
        }
    }

    /// The device's token from Apple's push service, once it has one.
    private(set) var token: String?
    private var waiting: [UUID: CheckedContinuation<String, Error>] = [:]

    /// Registers again at each launch where notifications are allowed, as Apple advises: tokens can change.
    func refresh() {
        Task {
            let settings = await UNUserNotificationCenter.current().notificationSettings()
            if [.authorized, .provisional, .ephemeral].contains(settings.authorizationStatus) {
                UIApplication.shared.registerForRemoteNotifications()
            }
        }
    }

    /// Whether notifications are allowed ("granted", "denied", or "default" before asking), and the token if there is one.
    func status() async -> [String: Any] {
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        var result: [String: Any] = ["environment": AppConfig.pushEnvironment]
        switch settings.authorizationStatus {
        case .authorized, .provisional, .ephemeral: result["permission"] = "granted"
        case .denied: result["permission"] = "denied"
        default: result["permission"] = "default"
        }
        if let token { result["token"] = token }
        return result
    }

    /// Asks permission if it hasn't been given yet, then registers with Apple's push service. Returns the token.
    func enable() async throws -> String {
        let granted = try await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])
        guard granted else { throw Failure.denied }
        return try await withCheckedThrowingContinuation { continuation in
            let id = UUID()
            waiting[id] = continuation
            UIApplication.shared.registerForRemoteNotifications()
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: 20_000_000_000)
                self?.waiting.removeValue(forKey: id)?.resume(throwing: Failure.timedOut)
            }
        }
    }

    func didRegister(_ deviceToken: Data) {
        let token = deviceToken.map { String(format: "%02x", $0) }.joined()
        self.token = token
        finish(.success(token))
    }

    func didFail(_ error: Error) {
        finish(.failure(error))
    }

    private func finish(_ result: Result<String, Error>) {
        let waiting = self.waiting
        self.waiting = [:]
        for continuation in waiting.values { continuation.resume(with: result) }
    }
}
