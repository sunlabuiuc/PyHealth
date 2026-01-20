from pyhealth.models.bhc_to_avs import BHCToAVS

# Initialize the model
model = BHCToAVS()

# Example Brief Hospital Course (BHC) text with common clinical abbreviations generated synthetically via ChatGPT 5.1
bhc = (
    "Pt admitted with acute onset severe epigastric pain and hypotension. "
    "Labs notable for elevated lactate, WBC 18K, mild AST/ALT elevation, and Cr 1.4 (baseline 0.9). "
    "CT A/P w/ contrast demonstrated peripancreatic fat stranding c/w acute pancreatitis; "
    "no necrosis or peripancreatic fluid collection. "
    "Pt received aggressive IVFs, electrolyte repletion, IV analgesia, and NPO status initially. "
    "Serial abd exams remained benign with no rebound or guarding. "
    "BP stabilized, lactate downtrended, and pt tolerated ADAT to low-fat diet without recurrence of sx. "
    "Discharged in stable condition w/ instructions for GI f/u and outpatient CMP in 1 week."
)

# Generate a patient-friendly After-Visit Summary
print(model.predict(bhc))

# Expected output: A simplified, patient-friendly summary explaining the hospital stay without medical jargon.