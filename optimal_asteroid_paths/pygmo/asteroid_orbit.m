

%%
% Run this in MATLAB — saves vertices and faces as CSV files
% which numpy can read directly

load('C:\Users\RAJAT GUPTA\Downloads\CnmSnm_asteroids.mat')   % change filename to match yours

% Check what variables loaded
whos
%%
% ── Adjust variable names below to match what 'whos' shows ──────────
% Common names are: V, vertices, Vertices, F, faces, Faces

vertices = themis_vertices;   % 402x3 — change 'V' to your actual variable name
faces    = themis_faces;   % 800x3 — change 'F' to your actual variable name
vertices = themis_vertices * 1000.0;  % # km → m  (only if needed)
% Save as CSV (zero-indexed faces for Python, MATLAB is 1-indexed)
writematrix(vertices, 'C:\Users\RAJAT GUPTA\Downloads\themis_vertices.csv');
writematrix(faces - 1, 'C:\Users\RAJAT GUPTA\Downloads\themis_faces.csv');   % subtract 1 for Python indexing

% Also save as .npz directly if you have MATLAB R2022b+ and Python linked
% Otherwise CSV is fine

disp('Saved themis_vertices.csv and themis_faces.csv')
disp(['Vertices: ' num2str(size(vertices))])
disp(['Faces:    ' num2str(size(faces))])