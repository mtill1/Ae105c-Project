def get_id_from_asteroid_name(asteroid_list, name):
    for asteroid in asteroid_list:
        if asteroid['NAME'] == name:
            return asteroid['ID']
    return -1
