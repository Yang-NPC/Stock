import datetime



# Example Unix timestamp
unix_timestamp = 1638334800

# Convert to datetime object
human_readable_date = datetime.datetime.fromtimestamp(unix_timestamp)

print(human_readable_date)

# Example human-readable date (e.g., "2023-12-03 15:00:00")
human_readable_date = "2023-12-03 15:00:00"

# Convert string to datetime object
date_time_obj = datetime.datetime.strptime(human_readable_date, '%Y-%m-%d %H:%M:%S')

# Convert datetime object to Unix timestamp
unix_timestamp = int(date_time_obj.timestamp())

human_readable_date_2 = datetime.datetime.fromtimestamp(unix_timestamp)

print(unix_timestamp)
print(human_readable_date_2)


seven_day = 7*24*60*60