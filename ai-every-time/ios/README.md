# The iPhone and iPad app

AI Every Time's app for iPhone and iPad shows the site (`https://www.aieverytime.org`) and adds what only an app can:
notifications through Apple's push service, the share sheet, actions on the app's icon, Siri and Spotlight
shortcuts, links off the site in Safari's viewer, downloads, and an offline screen that tries again when the
connection is back. Because the pages are the site's own, a change to the site shows in the app straight away, with
no new build.

People can get AI Every Time on an iPhone or iPad in three ways:

| | How | What you do |
|---|---|---|
| **The App Store** | Anyone finds and installs the app | Steps 1 to 7 below; Apple reviews it first |
| **TestFlight** | People you invite, by email or a public link, try the app before it's out | Steps 1 to 6 |
| **The Home Screen** | People add the site to their Home Screen from Safari. It opens full screen, like an app, with notifications | Nothing: it works today, and the site shows iPhone and iPad users how |

You don't need a Mac. GitHub builds, signs and uploads the app on its own Macs, from a button on GitHub. You need:

- **The Apple Developer Program**, $99 a year: [developer.apple.com/programs/enroll](https://developer.apple.com/programs/enroll/).
  Enrolling as an organisation shows its name as the seller on the App Store (it needs a D-U-N-S number); as an
  individual, your own name shows.
- **This GitHub repository**, where the workflow runs, and **the site on Render**, which the app shows.

Keys and passwords go only where these steps say: into GitHub's or Render's secret settings. Never into a chat, an
email or a file in the repository.

## 1. An App Store Connect API key, for GitHub

The workflow uses this key to register the app with Apple, get its signing certificate and upload builds.

1. In [App Store Connect](https://appstoreconnect.apple.com), open **Users and Access**, then **Integrations**, then
   **App Store Connect API**, and make sure **Team Keys** is selected.
2. Choose **Generate API Key** (or **+**). Name it `GitHub release` and give it the **App Manager** role.
3. Note the **Issuer ID** shown above the list and the new key's **Key ID**, then **Download** the key: a file named
   `AuthKey_` and the Key ID, ending `.p8`. Apple lets you download it only once, so keep it somewhere safe.
4. On GitHub, open this repository's **Settings**, then **Secrets and variables**, **Actions**, and add three
   **repository secrets** with **New repository secret**:

   | Name | Value |
   |---|---|
   | `APP_STORE_CONNECT_ISSUER_ID` | The Issuer ID |
   | `APP_STORE_CONNECT_KEY_ID` | The Key ID |
   | `APP_STORE_CONNECT_PRIVATE_KEY` | The whole `.p8` file: open it in a text editor and copy everything, the `-----BEGIN PRIVATE KEY-----` and `-----END PRIVATE KEY-----` lines included |

The workflow works out the private key for the app's signing certificate from this API key (see
`release/signing.py`), so there's nothing else to make or keep.

## 2. Register the app with Apple

1. On GitHub, open **Actions**, choose **iOS release** on the left, then **Run workflow**, leave the boxes empty and
   choose **Run workflow** again.
2. The first time, it registers the app ID `org.aieverytime.app` with Push Notifications, then stops with
   **Add the app in App Store Connect**. That's expected: the app needs a record in App Store Connect before builds
   can be uploaded, and that can only be made on Apple's website.

## 3. Add the app in App Store Connect

In App Store Connect open **Apps**, then **+**, **New App**, and fill in:

| Field | Value |
|---|---|
| Platforms | iOS |
| Name | `AI Every Time` (names are unique across the App Store; if it's taken, try `AI Every Time: Learn AI`) |
| Primary language | English (U.K.) |
| Bundle ID | `org.aieverytime.app` (in the list since step 2) |
| SKU | `aieverytime-ios` (yours only; never shown) |
| User access | Full access |

## 4. Upload a build

Run **iOS release** again (**Actions**, **iOS release**, **Run workflow**). It takes about 10 minutes, and when it
finishes, its summary says what happens next. Apple then processes the build, usually in 5 to 30 minutes, and emails
you when it's ready. The boxes are optional:

- **Version**: the version people see, such as `1.0`. Leave it blank to keep the project's (1.0). Use a higher one,
  such as `1.1`, for each new release on the App Store.
- **What to test**: a note shown to TestFlight testers.
- **Outside testers**: the name of a TestFlight group of outside testers (step 5). The build is sent to them once
  Apple's beta review approves it.

Each run gets a build number of its own, so you can upload as often as you like. A new build is only needed when the
app itself changes (this folder): changes to the site reach the app without one.

## 5. Test it with TestFlight

Testers install Apple's free **TestFlight** app from the App Store, then AI Every Time from it.

- **Your team.** In App Store Connect, open the app, then **TestFlight**, and under **Internal Testing** choose **+** to
  make a group, such as `Team`. Add people who are in your App Store Connect team (up to 100), and keep **Enable
  automatic distribution** on so every new build reaches them. They get an email invitation.
- **Anyone, with a link.** Under **External Testing** choose **+** and make a group named `Public beta`. Add the build,
  fill in **Test Information** (what to test, a feedback email, the privacy policy at
  `https://www.aieverytime.org/privacy`, and a sign-in for Apple's beta review, as in [AppStore.md](AppStore.md)),
  and submit it for review. The first build of each version is reviewed, usually within a day. Then choose **Enable
  Public Link** and copy the link (`https://testflight.apple.com/join/…`). Anyone with it can join, up to 10,000
  testers.
  - On Render, set `AIET_IOS_TESTFLIGHT_URL` to the link, and the site offers the beta to iPhone and iPad users.
  - From then on, type `Public beta` in **Outside testers** when you run **iOS release**, and new builds go to the
    group by themselves.

## 6. Notifications in the app

The app gets notifications through Apple's push service. Your server needs a key to send them:

1. At [developer.apple.com](https://developer.apple.com/account), open **Certificates, IDs & Profiles**, then
   **Keys**, and **+**. Name it `AI Every Time push` and tick **Apple Push Notifications service (APNs)**. If it asks,
   choose **Sandbox & Production** and **Team Scoped (All Topics)**. Then **Continue** and **Register**.
2. **Download** the key (another `.p8` file, again only once) and note its **Key ID**. Your **Team ID** is under
   **Membership details** on the same site.
3. On Render, open the service, then **Environment**, and add:

   | Name | Value |
   |---|---|
   | `AIET_APNS_KEY` | The whole `.p8` file, as in step 1 |
   | `AIET_APNS_KEY_ID` | Its Key ID |
   | `AIET_APNS_TEAM_ID` | Your Team ID |

   Save, and Render restarts the site with them. In the app, open **Profile**, turn notifications on and **Send a
   test**. Until these are set, the app's profile says notifications aren't set up on the server yet.

## 7. Send it to the App Store

1. In App Store Connect, open the app's version (**1.0 Prepare for Submission**) and fill it in from
   [AppStore.md](AppStore.md): the description, keywords, support and privacy addresses, category, the age rating
   questions, and **App Privacy**. Add the screenshots in `screenshots/`: those for 6.9-inch iPhones and 13-inch
   iPads.
2. Under **App Review Information**, give Apple a sign-in to try everything: make an account on the site, then, as an
   admin, give it a plan on the **People** tab of the Staff page. Paste the notes from [AppStore.md](AppStore.md).
3. Under **Build**, choose the build you uploaded, then **Add for Review** and **Submit for Review**. Apple usually
   answers in a day or two. Choose whether it goes live as soon as it's approved or when you release it.
4. Once it's live, copy the app's **Apple ID** (digits) from **App Information** and set `AIET_IOS_APP_ID` to it on
   Render. The site then links to the App Store, and Safari shows a banner for the app on the site. You can remove
   `AIET_IOS_TESTFLIGHT_URL` then.

### Paying for plans

Nothing is sold in the app: as Apple's rules ask, it shows no prices, upgrade buttons or links to pay, and learners
use the plan they already have. Everyone can use AI Every Time free, with limits.

Apple's guideline 3.1.3(b) lets apps on several platforms open what people bought elsewhere, but asks that it can be
bought in the app too, with Apple's in-app purchase. Apple's reviewers apply this unevenly, so the app may be approved
as it is, or Apple may ask for in-app purchase. If they do, the answer is to add Apple's in-app purchase for the plans
alongside the website's (Apple keeps 15% to 30% of those sales), or to reply explaining how the app is used. The
notes in [AppStore.md](AppStore.md) say how the app handles plans.

## Updates

- **The site:** deploy it as usual. The app shows the new pages straight away.
- **The app:** run **iOS release** with a higher **Version**, then in App Store Connect add the new version, choose
  its build and submit it, as in step 7.

## If something goes wrong

| What the workflow says | What to do |
|---|---|
| Apple secrets missing | Add the three secrets in step 1 |
| Add the app in App Store Connect | Do step 3, then run it again |
| Can't create a certificate: you have reached the maximum, or one is pending | At developer.apple.com, **Certificates**, revoke an Apple Distribution certificate you don't use, then run it again |
| FORBIDDEN, or the key isn't allowed to do something | Make a new key with the **Admin** role, and update the three secrets |
| The bundle version must be higher than the previous one | Run it again: each run has a higher build number |
| A version's train is closed, or the version already exists | That version is on the App Store already: type a higher one in **Version** |
| An identifier `org.aieverytime.app` isn't available | Someone else has it: change `PRODUCT_BUNDLE_IDENTIFIER` in `AIEveryTime.xcodeproj`, `BUNDLE_ID` in `.github/workflows/ios-release.yml` and `AIET_APNS_BUNDLE_ID` on Render to one of your own |

If you make a new API key, Apple gives the workflow a new signing certificate. The old one can then be revoked.

## For developers

- `AIEveryTime/`: the SwiftUI app (iOS 16 and later, iPhone and iPad). `WebController.swift` holds the web view and
  its rules (which links stay in the app, downloads, the system's alerts); `Bridge.swift` answers the page's
  questions (notifications, the share sheet, Settings) through the `aiet` message handler; `PushManager.swift`
  registers with Apple's push service; `Shortcuts.swift` has the actions on the icon and the Siri and Spotlight
  shortcuts. The site knows it's in the app from its user agent (`AIEveryTimeApp/<version>`) and leaves anything for
  sale out.
- `Info.plist` and `AIEveryTime.entitlements`: the usage descriptions, launch screen, icon actions and Push
  Notifications. `AIEveryTime/PrivacyInfo.xcprivacy` is the privacy manifest; keep it matching the answers in
  [AppStore.md](AppStore.md).
- The site the app shows is the `AIET_SITE_URL` build setting. With a Mac, open `AIEveryTime.xcodeproj` in Xcode 26
  or later, choose your team under **Signing & Capabilities**, and run it on a simulator or a phone; point
  `AIET_SITE_URL` at a test server to try the site's changes in the app first.
- `.github/workflows/ios.yml` builds the app on a Mac whenever it changes, and checks the release tools.
  `.github/workflows/ios-release.yml` releases it: Codemagic's open-source tools (`codemagic-cli-tools`, pinned)
  talk to Apple, and `release/signing.py` works out the certificate key, picks the App Store profile, writes the
  export options and checks the signed app before it's uploaded.
- The icon and launch logo come from the site's logo: `python scripts/make_icons.py`.
