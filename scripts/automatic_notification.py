import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from crontabs import Cron, Tab
from datetime import datetime


def send_email(sender_addr, receiver_addr, body, link):
#Function for sending email
  try:
    #Timestamp in email
    now = datetime.now()
    timestamp_string = now.strftime("%d-%b-%Y %H:%M:%S")

    #msg formation
    msg = MIMEMultipart()
    msg['From'] = sender_addr
    msg['To'] = receiver_addr
    subject = 'Cam2 Embedded Vision 2020 Video Data ' + timestamp_string
    msg['subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    msg.attach(link)
    text = msg.as_string()

    #server connection and sending email
    server = smtplib.SMTP('smtp.office365.com', 587)
    server.ehlo()
    server.starttls()         #secure connection
    server.login(email, password)

    server.sendmail(sender_addr, receiver_addr, text)
    server.quit()
    print('Email successfully sent.')
  except:
    print('Email failed to be sent.')

#information about sender and receiver
sender_addr = ""
receiver_addr = ""
email=""
password=""

#msg information
body="Hello. These are today's video data."
link = MIMEText(u'<a href="www.google.com">This is the link to the data</a>', 'html')

#Scheduling the running of program using crontabs
Cron().schedule(
  Tab(
    name='send_email'
  ).run(
    send_email, sender_addr, receiver_addr, body, link
 ).every(
    day=1
 ).starting(
    '08/15/2020 08:00'
  ).until(
    '11/20/2020 08:00'
  )
).go()

#Credit to the creator of crontabs
#The MIT License (MIT)

#Copyright (c) 2017 Rob deCarvalho

#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in
#the Software without restriction, including without limitation the rights to
#use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
#the Software, and to permit persons to whom the Software is furnished to do so,
#subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
#FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
