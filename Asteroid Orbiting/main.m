%% The Main Entry Point Into Asteroid Orbiter
%       _____            __                      .__    .___ ________       ___.   .__  __                
%      /  _  \   _______/  |_  ___________  ____ |__| __| _/ \_____  \______\_ |__ |__|/  |_  ___________ 
%     /  /_\  \ /  ___/\   __\/ __ \_  __ \/  _ \|  |/ __ |   /   |   \_  __ \ __ \|  \   __\/ __ \_  __ \
%    /    |    \\___ \  |  | \  ___/|  | \(  <_> )  / /_/ |  /    |    \  | \/ \_\ \  ||  | \  ___/|  | \/
%    \____|__  /____  > |__|  \___  >__|   \____/|__\____ |  \_______  /__|  |___  /__||__|  \___  >__|   
%            \/     \/            \/                     \/          \/          \/              \/       
% Essentially this establishes the CLI tool


clc;
load("core\title_card.mat", "TITLE");
for i = 1:length(TITLE)
    fprintf("%s\n", TITLE(i));
end
fprintf("Calculates, Plots, Exports, Simulates all things related to the Ae105C Asteroid Orbits.\n")
fprintf("(A Ae105C Spaghetti Code Mess by Trajectory and Operations Team.)\n\n")
clear TITLE;

fprintf("Loading Kernel Data From kernel_data.mat. Be sure to add all folders and subfolders in this directory to your MATLAB path.\n");
load("kernel_data.mat", "path_to_mice", "path_to_generic_kernels");

fprintf("Generic Kernel Path: %s\n Mice Path: %s\n", path_to_generic_kernels, path_to_mice);

usr_in = input("Is this correct? (y/n):", 's');

if usr_in ~= 'y' && usr_in ~= 'Y'
    path_to_generic_kernels = convertStringsToChars(input("Paste the Generic Kernel Path:", 's'));
    path_to_mice = convertStringsToChars(input("Paste the MICE Path:", 's'));
end

save("kernel_data.mat", "path_to_mice", "path_to_generic_kernels")

asteroid_list = load_kernels("bsp_files", path_to_mice, path_to_generic_kernels);
fprintf("Kernel data loaded successfully.\n\n\n");

fprintf("1) Simulate Orbit Equations Given Initial Conditions (NO CONTROL)\n")
fprintf("2) Plot Previously Calculated Solution\n")
fprintf("3) Optimize for Parameters (TODO)\n")
fprintf("4) Change Default Parameters\n")



mode_select = input("Which mode would you like to select? (1-4):");
while ~ismember(mode_select, 1:4)
    mode_select = input("Invalid selection. Please choose (1-4):");
end


switch mode_select
    case 1
        fprintf("\n\nSimulation selected. Listing available asteroids...\n");
        % Simulate Rendezvous Orbit
        for i = 1:length(asteroid_list)
            fprintf("#%3d | %10d | %s\n", i, asteroid_list(i).ID, asteroid_list(i).NAME);
        end

        aster_id = input(sprintf("Enter Asteroid Index (1-%d):", length(asteroid_list)));
        
        load("defaults.mat", "def_t_0_str", "def_t_f_str", "def_init_con_no_sim");

        initial_conditions = input("Enter initial conditions as a vector (press enter for default)(km & km/s) [x0, y0, z0, vx0, vy0, vz0]:");
        if isempty(initial_conditions)
            initial_conditions = def_init_con_no_sim;
        end

        t_0_str = input(sprintf("Enter initial date using Calendar Format (press enter for default: %s):", def_t_0_str), "s");
        if t_0_str == ""
            t_0_str = def_t_0_str;
        end
        et_0 = cspice_str2et(convertStringsToChars(t_0_str));

        t_f_str = input(sprintf("Enter final date using Calendar Format (press enter for default: %s):", def_t_f_str), "s");
        if t_f_str == ""
            t_f_str = def_t_f_str;
        end
        et_f = cspice_str2et(convertStringsToChars(t_f_str));
        
        clear def_t_f_str def_t_0_str def_init_con_no_sim;
        [t_sol, orbit_sol] = orbit_no_control(asteroid_list(aster_id), initial_conditions, et_0, et_f);
        
        fprintf("\n\nSimulation completed successfully. To what file will this data be saved?\n");

        save_filename = string(datetime("today"));
        usr_in = input(sprintf("File Name (press enter for default -> %s_%s.mat):", save_filename, asteroid_list(aster_id).NAME), "s");

        if usr_in ~= ""
            save_filename = usr_in;
        end

        save(sprintf("sim_data\\%s_%s.mat", save_filename, asteroid_list(aster_id).NAME), "t_sol", "orbit_sol");

        fprintf("\n\nCleaning up...\n");
        clear aster_id t_0_str t_f_str et_0 et_f initial_conditions i mode_select path_to_generic_kernels path_to_mice usr_in;
        %cspice_kclear;
        fprintf("Cleanup successful.\n");
    case 2
        fprintf("\n\nPlotting selected. Plotting imported data.\n");
        plot_orbit(t_sol, orbit_sol, 570/2, 550/2, 446/2, 50);
    case 3
        % Optimize for Parameters
        optimizeParameters();
    case 4
        % Optimize for Parameters
        optimizeParameters();
end



