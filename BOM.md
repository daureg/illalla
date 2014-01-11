name | what | when | how | format
tag_support | count, user and time for all tags above threshold | 6.12.13 | extract_dataset.supported_tags(DB, 20, 2, 0) | dict indexed by tag
tag_support.txt | text version of the previous | 6.12.13 | tag_support.py | plain text
out.patch | diff after tag cleaning | 6.11.13 | compare_tags.py | patch format
supported | original version of tag_support | 11.11.13 | extract_dataset.supported_tags(DB, 150, 25, 500) | dict indexed by tag
user_status | user and tourist status in SF | 31.10.13 | more_query.get_user_status(with_count=False) | dict id:bool
user_status_full | user and (photo count, tourist status) in SF | 3.11.13 | more_query.get_user_status(with_count=True) | dict id:(int,bool)
time_entropy.ods/txt | tags and their time entropy in SF | 7.11.13 | more_query.time_entropy() | plain text
