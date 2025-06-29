import os

# Set up your secret key, database URI, etc.
class Config:
    SECRET_KEY = os.urandom(24)  # To be used for securing sessions and cookies
    # DATABASE_URI = 'sqlite:///site.db'  # Example for a database URI
 
