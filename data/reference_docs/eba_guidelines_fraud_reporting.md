# European Banking Authority (EBA) Guidelines on Fraud Reporting under PSD2

**Issuing Body:** European Banking Authority (EBA)  
**Publication Date:** 2020-01-22  
**Version:** 1.2  
**Source URL:** https://www.eba.europa.eu/regulation-and-policy/payment-services-and-electronic-money/guidelines-on-fraud-reporting-under-psd2  
**Permitted Use:** Public regulatory standard and technical reporting guidelines.

---

## Section 1: Scope and Categorization of Payment Fraud
Under the Revised Payment Services Directive (PSD2) and EBA Guidelines (EBA/GL/2018/05 as amended), payment service providers (PSPs) must report statistical data on payment fraud across payment instruments, credit transfers, direct debits, card payments, and electronic money.

Fraudulent payment transactions are categorized under two primary legal mechanisms:
1. Unauthorized Payment Transactions: Transactions where the payment order was executed without the genuine consent or authorization of the payer (e.g., lost or stolen payment instruments, credential skimming, account takeover, card cloning).
2. Manipulation of the Payer (Authorized Push Payment / APP Fraud): Transactions where the payer was deceived or manipulated by a fraudster to issue a payment order to an account not controlled by the intended payee (social engineering, impersonation scams).

---

## Section 2: Card-Not-Present (CNP) Fraud and Velocity Controls
Card-Not-Present (CNP) fraud represents the predominant share of electronic card payment losses. Payment security frameworks must assess risk based on real-time transaction monitoring mechanisms:
- Deviations from typical spending habits: Unusual transaction monetary amounts exceeding established historical customer percentiles.
- High velocity of payment attempts: Multiple rapid successive payment authorizations at e-commerce merchants within minutes.
- Suspicious device or software environments: Changes in user-agent string, operating system version, or geolocation inconsistent with the authenticated payer's behavioral profile.
- Compromised merchant tokens: Clustering of fraudulent authorizations originating from merchant identifiers associated with known card credentials breaches.

---

## Section 3: Risk-Based Exemptions and Transaction Monitoring
PSPs applying exemptions from Strong Customer Authentication (SCA) under Transaction Risk Analysis (TRA) must maintain overall fraud rates below the regulatory threshold (e.g., 0.01% to 0.06% depending on transaction value).

Transaction monitoring systems must incorporate:
- Automated anomaly detection algorithms analyzing spending velocity, peer group spending averages, and geolocation markers.
- Scoring models that flag outlier transactions for immediate human analyst verification before settlement.
- Separation of risk scoring from definitive accusation: An elevated anomaly score mandates operational hold and customer outreach, not immediate asset forfeiture without corroborating factual evidence.

