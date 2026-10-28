import SwiftUI
import WebKit

/// The app's one screen: the site, a thin progress bar while a page loads, the launch cover until the first page is
/// ready, and the offline screen while the site can't be reached.
struct ContentView: View {
    @ObservedObject var web: WebController

    var body: some View {
        ZStack(alignment: .top) {
            Color("LaunchBackground").ignoresSafeArea()
            WebView(webView: web.webView).ignoresSafeArea()
            if web.started && web.isLoading {
                ProgressView(value: web.progress)
                    .progressViewStyle(.linear)
                    .tint(.accentColor)
            }
            if !web.started {
                LaunchCover().transition(.opacity)
            }
            if let failure = web.failure {
                OfflineView(detail: failure, retry: web.retry).transition(.opacity)
            }
        }
        .animation(.easeOut(duration: 0.25), value: web.started)
        .animation(.easeOut(duration: 0.2), value: web.failure)
    }
}

/// The web view, placed in SwiftUI. WebController makes it once and keeps it for the app's life.
struct WebView: UIViewRepresentable {
    let webView: WKWebView

    func makeUIView(context: Context) -> WKWebView { webView }

    func updateUIView(_ uiView: WKWebView, context: Context) {}
}

/// What shows while the first page loads: the launch screen's logo on its background, so the two run together, and a
/// spinner if it takes a while.
struct LaunchCover: View {
    @State private var slow = false

    var body: some View {
        ZStack {
            Color("LaunchBackground")
            Image("LaunchLogo")
            ProgressView()
                .offset(y: 120)
                .opacity(slow ? 1 : 0)
        }
        .ignoresSafeArea()
        .task {
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            withAnimation { slow = true }
        }
    }
}
