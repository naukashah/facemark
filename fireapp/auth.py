from pyrebase import pyrebase

from facemark.fireapp.views import config


def connect_firebase():
    # add a way to encrypt those, I'm a starter myself and don't know how
    username: "usernameyoucreatedatfirebase"
    password: "passwordforaboveuser"

    firebase = pyrebase.initialize_app(config)
    auth = firebase.auth()

    # authenticate a user > descobrir como não deixar hardcoded
    user = auth.sign_in_with_email_and_password(username, password)

    # user['idToken']
    # At pyrebase's git the author said the token expires every 1 hour, so it's needed to refresh it
    user = auth.refresh(user['refreshToken'])

    # set database
    db = firebase.database()

    return db
