# coding:utf -8

import smtplib  # smtp服务器
from email.mime.text import MIMEText  # 邮件文本


def send_mail(title, content):
    subject = f"{title} 任务完成"  # 邮件标题
    sender = ""  # 发送方
    recver = ""  # 接收方
    password = "EGHUPTWUVIZHLNLS"
    # 邮箱密码
    message = MIMEText(content, "plain", "utf-8")
    # content 发送内容     "plain"文本格式   utf-8 编码格式

    message['Subject'] = subject  # 邮件标题
    message['To'] = recver  # 收件人
    message['From'] = sender  # 发件人

    smtp = smtplib.SMTP_SSL("smtp.163.com", 994)  # 实例化smtp服务器
    smtp.login(sender, password)  # 发件人登录
    smtp.sendmail(sender, [recver], message.as_string())  # as_string 对 message 的消息进行了封装
