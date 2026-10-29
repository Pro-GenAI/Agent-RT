# Deployment privacy and launch checklist

Agent RT is a self-hosted library and runtime. Its public documentation site is a separate, static GitHub Pages site. A service built with Agent RT has its own operator, users, providers, storage, and legal obligations; the website's privacy notice does not cover that service.

## Current project boundaries

- The public site has no signup, contact form, newsletter, checkout, subscriptions, comments, or visitor uploads. It loads local assets and has no analytics or session replay. Its Motion preference stays in browser local storage. GitHub Pages logs visitor IP addresses for security ([GitHub documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages#data-collection)).
- Running Agent RT can send prompts, tool results, files, and retrieved context to configured model providers, memory services, MCP servers, and tools. Sessions, checkpoints, and traces can retain that content. Review the actual configuration and integrations; local execution does not mean every data flow stays local.
- OpenLLMetry export requires explicit configuration and an installed optional SDK. Review content capture, destinations, SDK telemetry, redaction, access controls, and retention before enabling it. Runtime tracing is distinct from recording a website visitor's interaction.
- GitHub issues, discussions, package registries, and linked services operate under their own policies. Do not submit secrets, children's information, private transcripts, or other sensitive data in public issues.

## Before adding a hosted product feature

Use this checklist when the relevant feature is introduced. Record the operator, audience, jurisdictions, data destinations, retention/deletion process, and reviewer before launch. These are engineering review points, not a certification of legal compliance.

| Area | Present project | Required implementation/review before launch |
| --- | --- | --- |
| Children and signup | The site provides developer documentation and has no registration or submission flow. | Determine the intended audience and whether COPPA or other children's privacy rules apply before collecting information. General-audience services are not universally required to ask age. If age screening is appropriate, put a neutral screen before personal-data collection and enforce the result on the server. Do not retain an underage submission or rely on a checkbox as parental consent. A child-directed service needs a reviewed parental notice/consent flow, data minimization, deletion, and retention controls. |
| Fonts and assets | Fonts use installed/system fallbacks; scripts, styles, icons, and images are served from the site. | Keep assets local and preserve their licenses. Review CSS imports, preload/preconnect, embedded media, and SDK requests as well as HTML URLs. Review disclosures and any necessary consent before introducing an external data recipient. |
| Analytics and session replay | No analytics, replay SDKs, or keystroke recording. | Keep recording disabled by default. Before any rollout, review applicable consent requirements and disclose the purpose and recipients. Where consent is required, load the SDK only after opt-in and stop collection on withdrawal. Mask inputs and exclude prompts, credentials, payment data, and private content from capture and export. |
| Marketing email | No mailing list or email campaign sender. | Use accurate sender/subject information and ad identification where required, a real valid postal address, and a working unsubscribe mechanism. Test suppression on every marketing send, honor opt-outs within the applicable deadline, and keep promotional content separate from operational mail. Do not publish a placeholder address or a cosmetic unsubscribe link. |
| Recurring billing | Free MIT software and documentation; no billing. | Before purchase, show price, currency, billing interval, trial-to-paid date, renewal terms, and cancellation instructions next to the purchase action. Obtain and record affirmative consent to recurring charges, provide required confirmations/reminders, and test accessible online cancellation end to end. Hiding terms in a policy page is insufficient. |
| User content and copyright | No visitor content hosting or upload endpoint on the public site. | Before hosting user submissions, define rights/permission rules, reporting and takedown procedures, a counter-notice process, and a repeat-infringer policy. If seeking US DMCA safe-harbor protection, assess eligibility and other conditions, designate and register an agent, publish matching contact details, and maintain the registration. A contact link alone is not an agent registration or safe harbor. |

## Maintainer checks

Run from the repository root after website changes:

```bash
python3 scripts/check_website_csp.py
python3 -m unittest discover -s scripts -p 'test_check_website_csp.py'
```

The gate checks loaded HTML asset URLs, inline-script hashes, and the restrictive CSP directives, including font and connection destinations and disabled form submissions. It rejects policy overrides or new directives until deliberately reviewed. It does not certify every JavaScript data flow, legal requirement, or deployed hosting configuration. Inspect new dependencies and browser requests as part of review; update the [website notice](website/docs/reference/privacy-terms.html), [website maintenance guide](website/README.md), and this document when behavior changes.

## Primary references

Reviewed 29 October 2026. Requirements depend on facts and jurisdiction; use the official guidance rather than social-media penalty estimates.

- [FTC COPPA FAQs](https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions), including the general-audience and age-screening guidance.
- [Google Fonts privacy FAQ](https://developers.google.com/fonts/faq/privacy), describing requests to Google's servers.
- [FTC CAN-SPAM compliance guide](https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business), including opt-out and postal-address requirements.
- [California Attorney General automatic-renewal guidance](https://www.oag.ca.gov/news/press-releases/attorney-general-bonta-issues-consumer-alert-california%E2%80%99s-automatic-renewal-law), including consent, reminders, and cancellation.
- [US Copyright Office DMCA directory FAQs](https://www.copyright.gov/dmca-directory/faq.html) and [section 512 overview](https://www.copyright.gov/512/), covering agent designation and other safe-harbor conditions.
