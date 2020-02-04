import pytest
import os
import requests

from handlers.web_api import WebAPI
from service.data_svc import DataService
from service.web_svc import WebService
from service.reg_svc import RegService
from service.ml_svc import MLService

from database.dao import Dao

dao = Dao(os.path.join('database', 'tram.db'))

web_svc = WebService()
reg_svc = RegService(dao=dao)
data_svc = DataService(dao=dao, web_svc=web_svc)
ml_svc = MLService(web_svc=web_svc, dao=dao)

website_to_test = ["https://www.fireeye.com/blog/threat-research/2016/11/fireeye_respondsto.html"]

@pytest.mark.asyncio
async def test_url_grab():
    test_data = '''In 2012, a suspected Iranian hacker group called the “Cutting Sword of Justice” used malware known as Shamoon – or Disttrack. In mid-November, Mandiant, a FireEye company, responded to the first Shamoon 2.0 incident against an organization located in the Gulf states. Since then, Mandiant has responded to multiple incidents at other organizations in the region.<br><br>Shamoon 2.0 is a reworked and updated version of the malware we saw in the 2012 incident. Analysis shows the malware contains embedded credentials, which suggests the attackers may have previously conducted targeted intrusions to harvest the necessary credentials before launching a subsequent attack.<br><br>FireEye HX and FireEye NX both detect Shamoon 2.0, and our Multi-Vector Virtual Execution (MVX) engine is also able to proactively detect this malware.<br><br>The following is a summary of what we know about Shamoon 2.0 based on the samples we’ve analyzed:<br><br>The malware scans the C-class subnet of the IP it has assigned to every interface on the system for target systems.<br><br>The malware then tries to access the ADMIN$, C$\\Windows, D$\\Windows, and E$\\Windows shares on the target systems with current privileges.<br><br>If current privileges aren’t enough to access the aforementioned shares, it uses hard coded, domain specific credentials (privileged credentials, likely Domain Administrator or local Administrator) gained during an earlier phase of the attack to attempt the same.<br><br>Once it has access, it enables the Remote Registry service on the target device and sets HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\System\\LocalAccountTokenFilterPolicy to 0 to enable share access.<br><br>Once it has performed the earlier actions, it copies ntssrvr32.exe to the %WINDIR%\\system32 of the target system and schedules an unnamed task (e.g. At1.job) to execute the malware.<br><br>The identified malware had a hard coded date to launch the wiping. Systems infected with the malware scheduled the job to start the process shortly thereafter.<br><br>The malware sets the system clock to a random date in August 2012. Analysis suggests this might be for the purposes of ensuring the component (a legitimate driver used maliciously) that wipes the Master Boot Record (MBR) and Volume Boot Record (VBR) is within its test license validity period.<br><br>While the original Shamoon malware attempted to overwrite operating system files with an image of a burning U.S. flag, the recently discovered variant attempts to overwrite Windows operating system files, although with a different image, a .JPG file depicting the death of Alan Kurdi, a Syrian child migrant who died while attempting to cross the Mediterranean Sea.<br><br>The following is guidance for detecting the malware, counteracting its activity, and attempting to prevent it from propagating in an environment. Please note that performing any of these actions could have a negative effect and should not be implemented without proper review and study of the impact of the environment.<br><br>Monitor any events in the SIEM that show dates in August 2012.<br><br>Monitor for system time change events that set the clock back to and from August 2012.<br><br>Monitor for Remote Registry service starts.<br><br>Monitor for changes to the aforementioned registry key value, if the value is currently non-zero.<br><br>Prevent and limit access to the aforementioned shares, which could have significant impact based on setup.<br><br>Prevent client-to-client communication to slow down the spread of the malware, which could also have a significant impact based on setup.<br><br>Monitor filesystems for the creation of any of the filenames provided in the Indicators of Compromise list at the bottom of the post.<br><br>Change the credentials of all privileged accounts and ensure local Administrator passwords are unique per system.<br><br>Indicators of Compromise<br><br>The following is a set of the Indicators of Compromise for the identified Shamoon variant. We recommend that critical infrastructure organizations and government agencies (especially those in the Gulf Cooperation Council region) check immediately for the presence or execution of these files within their Windows Server and Workstation environments. Additionally, we recommend that all customers continue to regularly review and test disaster recovery plans for critical systems within their environment.<br><br>File name: ntssrvr64.exe<br><br>Path: %SYSTEMROOT%\\System32<br><br>Compile Time: 2009/02/15 12:32:19<br><br>File size:717,312<br><br>File name: ntssrvr32.exe<br><br>Path: %SYSTEMROOT%\\System32 NA NA<br><br>File size: 1,349,632<br><br>File name: ntssrvr32.bat<br><br>Path: %SYSTEMROOT%\\System32 NA<br><br>MD5: 10de241bb7028788a8f278e27a4e335f<br><br>File size: 160<br><br>File name: gpget.exe<br><br>Path: %SYSTEMROOT%\\System32<br><br>PE compile time: 2009/02/15 12:30:41<br><br>MD5: c843046e54b755ec63ccb09d0a689674<br><br>File Size: 327,680<br><br>File name: drdisk.sys<br><br>Path: %SYSTEMROOT%\\System32\\Drivers<br><br>Compile time: 2011/12/28 16:51:29<br><br>MD5: 76c643ab29d497317085e5db8c799960<br><br>File Size: 31,632<br><br>File name: key8854321.pub<br><br>Path: %SYSTEMROOT%\\System32<br><br>MD5: b5d2a4d8ba015f3e89ade820c5840639 782<br><br>File name: netinit.exe<br><br>Path: %SYSTEMROOT%\\System32<br><br>MD5: ac4d91e919a3ef210a59acab0dbb9ab5<br><br>File Size: 183,808<br><br>Service Details<br><br>Display name: "Microsoft Network Realtime Inspection Service"<br><br>Service name: "NtsSrv"<br><br>Description: "Helps guard against time change attempts targeting known and newly discovered vulnerabilities in network time protocols"<br><br>Files created:<br><br>%WINDIR%\\inf\\usbvideo324.pnf<br><br>%WINDIR%\\system32<br><br>etinit.exe<br><br>Dynamic Analysis Observables<br><br>RegistryItem HKLM\\SYSTEM\\CurrentControlSet\\Services\\NtsSrv\\<br><br>RegistryItem HKLM\\SYSTEM\\ControlSet001\\Services\\NtsSrv\\<br><br>RegistryItem HKLM\\SYSTEM\\CurrentControlSet\\Services\\wow32\\<br><br>RegistryItem HKLM\\SYSTEM\\ControlSet001\\Services\\wow32\\<br><br>RegistryItem HKLM\\SYSTEM\\CurrentControlSet\\Services\\drdisk\\<br><br>RegistryItem HKLM\\SYSTEM\\ControlSet001\\Services\\drdisk\\<br><br>FileItem C:\\Windows\\System32\\caclsrv.exe<br><br>FileItem C:\\Windows\\System32\\certutl.exe<br><br>FileItem C:\\Windows\\System32\\clean.exe<br><br>FileItem C:\\Windows\\System32\\ctrl.exe<br><br>FileItem C:\\Windows\\System32\\dfrag.exe<br><br>FileItem C:\\Windows\\System32\\dnslookup.exe<br><br>FileItem C:\\Windows\\System32\\dvdquery.exe<br><br>FileItem C:\\Windows\\System32\\event.exe<br><br>FileItem C:\\Windows\\System32\\extract.exe<br><br>FileItem C:\\Windows\\System32\\findfile.exe<br><br>FileItem C:\\Windows\\System32\\fsutl.exe<br><br>FileItem C:\\Windows\\System32\\gpget.exe<br><br>FileItem C:\\Windows\\System32\\iissrv.exe<br><br>FileItem C:\\Windows\\System32\\ipsecure.exe<br><br>FileItem C:\\Windows\\System32\\msinit.exe<br><br>FileItem C:\\Windows\\System32<br><br>etx.exe<br><br>FileItem C:\\Windows\\System32<br><br>tdsutl.exe<br><br>FileItem C:\\Windows\\System32<br><br>tfrsutil.exe<br><br>FileItem C:\\Windows\\System32<br><br>tnw.exe<br><br>FileItem C:\\Windows\\System32\\power.exe<br><br>FileItem C:\\Windows\\System32\\rdsadmin.exe<br><br>FileItem C:\\Windows\\System32\\regsys.exe<br><br>FileItem C:\\Windows\\System32\\routeman.exe<br><br>FileItem C:\\Windows\\System32\\rrasrv.exe<br><br>FileItem C:\\Windows\\System32\\sacses.exe<br><br>FileItem C:\\Windows\\System32\\sfmsc.exe<br><br>FileItem C:\\Windows\\System32\\sigver.exe<br><br>FileItem C:\\Windows\\System32\\smbinit.exe<br><br>FileItem C:\\Windows\\System32\\wcscript.exe'''
    html_data = await web_svc.get_url("https://www.fireeye.com/blog/threat-research/2016/11/fireeye_respondsto.html")
    assert html_data == test_data

@pytest.mark.asyncio
async def test_remove_html_and_markup():
    test_data = '<html>This is an html page<p>Here is a paragraph</p><img src="and an image">Image title</img><m>and some random tags</m></html>'
    verification_data = 'This is an html pageHere is a paragraphImage titleand some random tags'
    assert await web_svc.remove_html_markup_and_found(test_data) == verification_data

@pytest.mark.asyncio
async def test_text_list_creator():
    #test_url = "http://x.com" # "https://www.webscraper.io/test-sites/e-commerce/allinone"
    text = "this is text\nthisismoretext\n\neven more text"
    validation_text = ["this is text","thisismoretext","even more text"]
    #r = requests.get(test_url)
    text_list = await web_svc._extract_text_as_list(text)
    assert text_list == validation_text

@pytest.mark.asyncio
async def test_html_list_creator():
    text = "<html>\n<p>This is a paragraph</p>\n<div>\n<p>this is a paragraph in a div</p>\n</div>\n</html>"
    validation_text = ['<html>', '<p>This is a paragraph</p>', '<div>', '<p>this is a paragraph in a div</p>', '</div>', '</html>']

    html_list = await web_svc._extract_html_as_list(text)
    assert html_list == validation_text

@pytest.mark.asyncio
async def test_build_dicts():
    e = {'uid':1,'text':'this is text','tag':'img','found_status':True}
    verify = {'uid': 1, 'text': 'this is text', 'tag': 'img', 'found_status': True, 'hits': None, 'confirmed': 'false'}
    test_out = await web_svc._build_final_image_dict(e)
    assert verify == test_out

    sent = {'uid':2,'found_status':True,'hits':'here','confirmed':True}
    verify2 = {'uid': 2, 'text': 'this is a single sentance', 'tag': 'p', 'found_status': True, 'hits': 'here', 'confirmed': True}
    test_out = await web_svc._build_final_html_text(sent,"this is a single sentance")
    assert verify2 == test_out


@pytest.mark.asyncio
async def test_html_mapping():
    test_html = web_svc.map_all_html(website_to_test[0]) 
