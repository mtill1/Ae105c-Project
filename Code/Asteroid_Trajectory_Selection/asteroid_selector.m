clear;

try
    asteroid_list = LOAD_KERNELS("NOTABLE_ASTEROID_BSPs");
catch exception
    close all force;
    fprintf('An error occurred: %s\n', exception.message);
    error("Most likely you need to add NOTABLE_ASTEROID_BSPs to your path" + ...
            " and you must be in the main github folder.");
end

MAX_M = 2;
MIN_M = 0;

MIN_LAUNCH_DATE_UTC = 'Jan 1 12:00:00 UTC 2027';
MAX_LAUNCH_DATE_UTC = 'Dec 31 12:00:00 UTC 2030';

for m_1 = MIN_M:MAX_M
    for m_2 = MIN_M:MAX_M
        for m_3 = MIN_M:MAX_M
            GENERATE_OPTIMIZED_DATA(asteroid_list, m_1, m_2, m_3, MIN_LAUNCH_DATE_UTC, MAX_LAUNCH_DATE_UTC);
        end
    end
end


