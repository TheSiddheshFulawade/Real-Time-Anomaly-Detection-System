from twilio.rest import Client

account_sid = 'ACe40d72023e80b216647a05d1f036502e'
auth_token = '9432b8bed0e024a3e2d0cd05a7b12446'
client = Client(account_sid, auth_token)

def make_twilio_call():
    try:
        call = client.calls.create(
            url='http://demo.twilio.com/docs/voice.xml',
            to='+917715935534',
            from_='+12055024489'
        )
        print(call.sid)
    except Exception as e:
        print("Twilio call failed:", e)

