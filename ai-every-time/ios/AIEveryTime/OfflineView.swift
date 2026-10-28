import SwiftUI

/// Shown while the site can't be reached, with a way to try again. The app also tries again by itself as soon as the
/// connection comes back (WebController.watchNetwork).
struct OfflineView: View {
    let detail: String
    let retry: () -> Void

    var body: some View {
        ZStack {
            Color("LaunchBackground").ignoresSafeArea()
            VStack(spacing: 16) {
                Image("LaunchLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 96, height: 96)
                Text("Can't reach AI Every Time")
                    .font(.title2.bold())
                    .multilineTextAlignment(.center)
                Text("Your feed, guides and companions need the internet. Check your connection, then try again.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                Text(detail)
                    .font(.footnote)
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
                Button(action: retry) {
                    Text("Try again").frame(minWidth: 160)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .padding(.top, 8)
            }
            .padding(32)
            .frame(maxWidth: 440)
        }
    }
}
