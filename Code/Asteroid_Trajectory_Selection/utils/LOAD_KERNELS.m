function asteroid_list = LOAD_KERNELS(BSP_FOLDER_NAME)
    % 1. IDENTIFY USER AND OPERATING SYSTEM
    [~, username] = system('whoami');
    username = strtrim(username); % Clean up any hidden spaces

    if contains(username, 'marissatill') || ismac
        % --- MARISSA'S MAC CONFIGURATION ---
        path_to_generic_kernels = '/Users/marissatill/Downloads/junior year/fall/ae105a/data';
        
        % so MATLAB can find the cspice functions.
        path_to_mice_mac = '/Users/marissatill/Downloads/junior year/fall/ae105a/mice'; 
        addpath(fullfile(path_to_mice_mac, 'src', 'mice'));
        addpath(fullfile(path_to_mice_mac, 'lib'));
        
        % Set filename for leapseconds (Mac usually doesn't use .pc)
        lsk_file = 'naif0012.tls';
    else
        % --- ORIGINAL WINDOWS CONFIGURATION ---
        path_to_generic_kernels = 'C:\Users\SPLog\OneDrive\Documents\Ae105A\kernels';
        path_to_mice            = 'C:\Users\SPLog\OneDrive\Documents\Ae105A\mice';
        
        addpath(fullfile(path_to_mice, 'src', 'mice'));
        addpath(fullfile(path_to_mice, 'lib'));
        
        % The original code used the .pc extension for Windows
        lsk_file = 'naif0012.tls.pc';
    end

    % 2. LOAD UNIVERSAL KERNELS
    % fullfile() automatically handles / for Mac and \ for Windows
    cspice_furnsh(fullfile(path_to_generic_kernels, 'lsk', lsk_file));
    cspice_furnsh(fullfile(path_to_generic_kernels, 'spk', 'satellites', 'jup310.bsp'));
    cspice_furnsh(fullfile(path_to_generic_kernels, 'spk', 'planets', 'de430.bsp'));
    cspice_furnsh(fullfile(path_to_generic_kernels, 'pck', 'gm_de431.tpc'));
    cspice_furnsh(fullfile(path_to_generic_kernels, 'pck', 'pck00010.tpc'));

    % 3. LOAD PROJECT-SPECIFIC KERNELS (Asteroids/Clipper)
    % Resolve BSP folders robustly. Note: MATLAB's isfolder/exist can return
    % true for folders that are merely on the MATLAB path; dir() still needs
    % an actual filesystem path. Prefer the repo-root layout:
    %   data/NOTABLE_ASTEROID_BSPs
    %   data/SPICE_BSPs
    bspFolder = '';
    candidate = fullfile('data', BSP_FOLDER_NAME);
    if isfolder(candidate)
        bspFolder = candidate;
    elseif isfolder(fullfile(pwd, BSP_FOLDER_NAME))
        bspFolder = BSP_FOLDER_NAME;
    else
        w = what(BSP_FOLDER_NAME);
        if ~isempty(w) && isfield(w, 'path') && isfolder(w.path)
            bspFolder = w.path;
        else
            bspFolder = BSP_FOLDER_NAME;
        end
    end

    % Find all .bsp files in the resolved folder
    bsp_files = dir(fullfile(bspFolder, '*.bsp'));

    if isempty(bsp_files)
        error("No .bsp files found in '%s'. Expected '%s' or '%s'.", ...
            bspFolder, BSP_FOLDER_NAME, fullfile('data',BSP_FOLDER_NAME));
    end
    
    % Pre-allocate the structure array
    asteroid_list(length(bsp_files)) = struct('ID', [], 'NAME', []);
    
    for i = 1:length(bsp_files)
        % Construct the full path to the specific .bsp file
        formatted_bsp_file = fullfile(bsp_files(i).folder, bsp_files(i).name);
        
        % Load it into the SPICE system
        cspice_furnsh(formatted_bsp_file);
        
        % Extract the ID from the SPK file
        [id] = cspice_spkobj(formatted_bsp_file, 1000);
        
        % Store results
        asteroid_list(i).ID = id;
        asteroid_list(i).NAME = bsp_files(i).name(1:end-4); % Remove '.bsp'
    end
end