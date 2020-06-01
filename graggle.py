from app import app

'''
# Configs for nginx
app.config.update({
# remove the default of '/'
'routes_pathname_prefix': '',

# remove the default of '/'
'requests_pathname_prefix': ''
})
'''

application = app.server

#if __name__ == "__main__":
#    app.run()
