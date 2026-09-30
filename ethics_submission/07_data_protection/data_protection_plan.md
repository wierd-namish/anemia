# Data Protection & Privacy Governance Plan

**Document Identifier:** ETH-SUB-DAT-001  
**Study Phase:** Phase 6 Clinical Validation  
**Date:** 2026-09-30  
**Statutory Framework:** Ghana Data Protection Act, 2012 (Act 843)  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. Principles of Data Processing

In adherence to Section 17 of the Ghana Data Protection Act 2012 (Act 843):
1. **Lawfulness & Transparency:** Data is collected solely for the approved research protocol following signed parental informed consent.
2. **Data Minimization:** Only isolated fingernail photographs and necessary study covariates (age in months, sex, blood Hb) are collected. No direct identifiers (names, home addresses, phone numbers) are attached to image payloads.
3. **Storage Limitation:** In standard screening mode, raw image pixels are processed in volatile RAM and immediately discarded. In the blinded prospective study archive, de-identified research images are stored with strict access controls.
4. **Integrity & Confidentiality:** Data at rest is encrypted using AES-256; data in transit is encrypted using TLS 1.3.

---

## 2. Participant Pseudonymization

* **Participant Token Format:** `GH-PML-001` (Site Prefix - Clinic Code - Serial Index).
* **Master Key Re-identification File:** A single encrypted password-protected spreadsheet linking patient medical record numbers to study UUIDs will be stored on a secure, air-gapped hospital workstation under the exclusive custody of the Principal Investigator.
