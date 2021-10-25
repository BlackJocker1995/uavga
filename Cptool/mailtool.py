# coding:utf -8

import smtplib  # smtp服务器
from email.mime.text import MIMEText  # 邮件文本


def send_mail(title, content):
    subject = f"{title} 任务完成"
    sender = ""  #
    recver = ""  #
    password = "EGHUPTWUVIZHLNLS"
    # 邮箱密码
    message = MIMEText(content, "plain", "utf-8")

    message['Subject'] = subject  #
    message['To'] = recver  #
    message['From'] = sender  #

    smtp = smtplib.SMTP_SSL("smtp.163.com", 994)  #
    smtp.login(sender, password)  # 发件人登录
    smtp.sendmail(sender, [recver], message.as_string())  #
