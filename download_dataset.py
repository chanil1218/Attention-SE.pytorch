#!/usr/bin/env python
# coding: utf-8

# In[49]:


from google_drive_downloader import GoogleDriveDownloader as gdd


# In[26]:


# Download Train set
gdd.download_file_from_google_drive(file_id='1CULVCAq0T3wqZTPGIqPja6OtwjYJkGAy',
                                    dest_path='dataset/train.tar.gz',
                                    unzip=False, showsize=True)
get_ipython().system('tar xvzf dataset/train.tar.gz')


# In[53]:


# Download Valid set
gdd.download_file_from_google_drive(file_id='1WE229Jt9WV2iZbxY7YjkYIfSZyCHz9Iq',
                                    dest_path='dataset/valid.tar.gz',
                                    unzip=False, showsize=True)
get_ipython().system('tar xvzf dataset/valid.tar.gz')


# In[ ]:


# Download Test set
gdd.download_file_from_google_drive(file_id='1mERBbcwBgGjRpHeGf3d871DEW4EvI7fi',
                                    dest_path='dataset/valid.tar.gz',
                                    unzip=False, showsize=True)
get_ipython().system('tar xvzf dataset/test.tar.gz')


# In[43]:


ls


# In[ ]:




