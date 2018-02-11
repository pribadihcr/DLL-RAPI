# add our app to the system path
import sys
sys.path.insert(0, "/var/www/html/DLL-RAPI")

import os.path as osp
# import the application and away we go...
#this_dir = osp.dirname(__file__)

#sys.path.append(osp.join(this_dir, 'API'))

from web_RAPI import app as application
