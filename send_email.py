import os
import smtplib
import imghdr
from email.message import EmailMessage

EMAIL_ADDRESS = os.environ.get('EMAIL_USER')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')

files = ['image1.jpg', 'image2.jpg']

contacts = ['contact1', 'contact2']

msg = EmailMessage()
msg['Subject'] = 'Good Morning!'
msg['From'] = EMAIL_ADDRESS
msg['To'] = contacts # or ', '.join(contacts) > read the documentation.

msg.set_content('This is a plain text email')
for file in files:
    with open('bronx.jpg', 'rb') as f:
        file_data = f.read()
        file_type = imghdr.what(f.name)
        file_name = f.name

    msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

msg.add_alternative("""\
<!DOCTYPE html>
<html>
    <body>
        <h1 style="color:SlateGray;">This is an HTML Email!</h1>
    </body>
</html>
""", subtype='html')


with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    smtp.send_message(msg)
