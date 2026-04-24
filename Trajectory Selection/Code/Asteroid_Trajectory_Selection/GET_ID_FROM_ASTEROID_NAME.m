function asteroid_id = GET_ID_FROM_ASTEROID_NAME(asteroid_list, name)
    formatted_name_list = [asteroid_list.NAME, "LAST"];
    formatted_name_list = formatted_name_list(1:end-1);
    
    asteroid_in_list = any(formatted_name_list == name);

    if asteroid_in_list
        asteroid_id = asteroid_list(formatted_name_list == name).ID;
    else
        asteroid_id = -1; 
    end
end