import json 

d = {
  "attack-pattern--00d0b012-8a03-410e-95de-5826bf542de6": {
    "description": "If a malicious tool is detected and quarantined or otherwise curtailed, an adversary may be able to determine why the malicious tool was detected (the indicator), modify the tool by removing the indicator, and use the updated version that is no longer detected by the target's defensive systems or subsequent targets that may use similar systems.\n\nA good example of this is when malware is detected with a file signature and quarantined by anti-virus software. An adversary who can determine that the malware was quarantined because of its file signature may use [Software Packing](https://attack.mitre.org/techniques/T1045) or otherwise modify the file so it has a different signature, and then re-use the malware.",
    "example_uses": [
      "The author of  submitted samples to VirusTotal for testing, showing that the author modified the code to try to hide the DDE object in a different part of the document.",
      "apparently altered  samples by adding four bytes of random letters in a likely attempt to change the file hashes.",
      "Find-AVSignature AntivirusBypass module can be used to locate single byte anti-virus signatures.",
      "has been known to remove indicators of compromise from tools.",
      "Analysis of  has shown that it regularly undergoes technical improvements to evade anti-virus detection.",
      "Based on comparison of  versions,  made an effort to obfuscate strings in the malware that could be used as IoCs, including the mutex name and named pipe.",
      "has tested malware samples to determine AV detection and subsequently modified the samples to ensure AV evasion.",
      "includes a capability to modify the \"beacon\" payload to eliminate known signatures or unpacking methods.",
      "has updated and modified its malware, resulting in different hash values that evade detection."
    ],
    "id": "T1066",
    "name": "Indicator Removal from Tools",
    "similar_words": [
      "Indicator Removal from Tools"
    ]
  }
}

for k in d.items():
    examples = k[1]['example_uses']
    [print(e) for e in examples]

# for k in d.items():
#     for i in k:
#         print(i)
#         print("OK")