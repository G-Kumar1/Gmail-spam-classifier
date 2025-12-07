import imaplib
import email
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

load_dotenv()
# ========== ✅ DIRECT CREDENTIAL FALLBACK (GUARANTEED WORKING) ==========
# ⚠️ PUT YOUR REAL VALUES HERE (TEMPORARILY FOR TESTING)

EMAIL_ACCOUNT = os.getenv("EMAIL_ACCOUNT" ) # <-- Your full Gmail address
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD" )  # <-- Your 16-digit APP PASSWORD (no spaces)


# ========== SAFETY CHECK ==========
if not EMAIL_ACCOUNT or not EMAIL_PASSWORD:
    raise ValueError("❌ EMAIL_ACCOUNT or EMAIL_PASSWORD is missing in the script.")


# ========== HTML CLEANER ==========
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


# ========== FETCH EMAILS ==========
def fetch_recent_emails(limit=20):
    mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)

    print("✅ Connecting to Gmail IMAP...")
    mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
    print("✅ Login successful!")

    mail.select("inbox")

    status, messages = mail.search(None, "ALL")

    if status != "OK":
        raise RuntimeError("❌ Could not read inbox.")

    email_ids = messages[0].split()

    if not email_ids:
        print("⚠️ Inbox is empty.")
        return []

    latest_ids = email_ids[-limit:]

    emails = []

    for e_id in latest_ids:
        status, msg_data = mail.fetch(e_id, "(RFC822)")

        if status != "OK":
            continue

        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        subject = msg.get("Subject", "")
        sender = msg.get("From", "")

        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                try:
                    payload = part.get_payload(decode=True)
                    if not payload:
                        continue

                    decoded = payload.decode(errors="ignore")
                except:
                    continue

                if content_type == "text/plain" and "attachment" not in content_disposition:
                    body = decoded
                    break

                elif content_type == "text/html" and not body:
                    body = clean_html(decoded)

        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode(errors="ignore")
            except:
                body = ""

        emails.append({
            "subject": subject,
            "from": sender,
            "body": body,
            "text": f"Subject: {subject}\n\n{body}"
        })

    mail.logout()
    return emails


# ========== TEST RUN ==========
if __name__ == "__main__":
    data = fetch_recent_emails(limit=5)

    for d in data:
        print("=" * 80)
        print("FROM:", d["from"])
        print("SUBJECT:", d["subject"])
        print(d["body"][:500])
