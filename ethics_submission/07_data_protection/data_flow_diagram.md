# Clinical Data Flow & Linkage Diagram

**Document Identifier:** ETH-SUB-DAT-002  
**Date:** 2026-09-30  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

```
[ PARTICIPANT VISIT ]
        │
        ├── Signed Parental Consent (Physical Paper Record -> Locked Hospital Archive)
        │
        ├── 1. MOBILE CAMERA ACQUISITION
        │      └── 3-4 Nail Photographs (UUID tagged, No Face/PII)
        │            │
        │            ▼ (TLS 1.3 Encryption)
        │      [ AI INFERENCE CONTAINER (In-Memory Processing) ]
        │            │
        │            ├── Inference Result -> Temporary Screen Display
        │            └── De-identified Image Archive -> Encrypted Study Server
        │
        └── 2. VENOUS PHLEBOTOMY (<15 min)
               └── 0.5 mL EDTA Blood Tube (Labeled with UUID)
                     │
                     ▼
               [ CENTRAL CLINICAL LABORATORY ]
                     │
                     ├── Sysmex XN-350/550 Analysis (SLS Method)
                     └── Reference Hb (g/dL) -> Blinded Lab Database
                           │
                           ▼ (Post-Database Lock Linkage)
               [ BLINDED INDEPENDENT STATISTICIAN ]
                     └── Paired SAP Accuracy Analysis (Sensitivity / Specificity)
```
