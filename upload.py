#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dropbox
import sys
import time
import os
import shutil
from dropbox.files import WriteMode

class TransferData:
    def __init__(self):
        self.access_token = '932ZpWKO_10AAAAAAAACe9ZO8K_FOFJz7bQyHi0k-VVbOmq50aCxCFt94y5zfpPI'

    def upload_file(self, file_from, folder_to=None, delete_original=False):
    	original_file = file_from
    	zip_created = False
    	if os.path.isdir(file_from):
    		print ('Converting folder to zip file...')
    		shutil.make_archive('tmp' + file_from, 'zip', file_from)
    		file_from = file_from + '.zip'
    		zip_created = True
    	if folder_to is None:
    		file_to = '/' + os.path.basename(file_from)
    	else:
    		file_to = os.path.join('/' + folder_to, os.path.basename(file_from))
    	if zip_created:
    		file_from = 'tmp' + file_from
    	print ('Uploading file from', file_from, 'to dropbox...')
    	dbx = dropbox.Dropbox(self.access_token)    	
    	file_size = os.path.getsize(file_from)
    	CHUNK_SIZE = 4 * 1024 * 1024
    	t0 = time.time()
    	for i in range(5):
    		try:
		    	if file_size <= CHUNK_SIZE:
		    		with open(file_from, 'rb') as f:
		    			dbx.files_upload(f.read(), file_to, mode=WriteMode.overwrite)
		    	else:
		    		f = open(file_from, 'rb')
		    		upload_session_start_result = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
		    		cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id, offset=f.tell())
		    		commit = dropbox.files.CommitInfo(path=file_to)
		    		while f.tell() < file_size:
		    			if ((file_size - f.tell()) <= CHUNK_SIZE):
		    				dbx.files_upload_session_finish(f.read(CHUNK_SIZE), cursor, commit)
		    			else:
		    				dbx.files_upload_session_append(f.read(CHUNK_SIZE), cursor.session_id, cursor.offset)
		    				cursor.offset = f.tell()
	    		t1 = time.time()
	    		print ('Finished uploading in', (t1-t0)/60, 'minutes')
	    		break
	    	except:
	    		e = sys.exc_info()[0]
	    		print ('faild to upload the file. here are the ditails of the error:')
	    		print (e)
	    		if i < 4:
		    		print ('trying', 4-i, 'more times')
	    		if i == 4:
	    			delete_original = False
    	if zip_created:
    		file_from = os.path.normpath(file_from)
    		file_from = file_from.split(os.sep)[0]
    		print ('Removing zip file...')
    		if os.path.isdir(file_from):
    			shutil.rmtree(file_from)
    		else:
    			os.remove(file_from)
    	if delete_original:
    		if os.path.isdir(original_file):
	    		shutil.rmtree(original_file)
	    	else:
	    		os.remove(original_file)

if __name__ == '__main__':
	file_from = sys.argv[1]
	delete = False
	file_to = None
	if len(sys.argv) > 2:
		if sys.argv[2] == 'delete':
			delete = True
		else:
			file_to = sys.argv[2]
			if len(sys.argv) == 4:
				if sys.argv[3] == 'delete':
					delete = True	
	transferData = TransferData()
	transferData.upload_file(file_from, file_to, delete_original=delete)
	

