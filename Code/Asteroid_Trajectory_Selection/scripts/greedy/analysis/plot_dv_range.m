LAUNCH_RANGE = [cspice_str2et('Jan 1 12:00:00 UTC 2028'),...
    cspice_str2et('Dec 31 12:00:00 UTC 2028')];

LAUNCH_BODY = int2str(asteroid_list(2).ID);
LANDING_BODY = "-1";
GOAL_BODY = int2str(asteroid_list(3).ID);
M_1 = -1;

MINUTE = 60;
HOUR = 60 * MINUTE;
DAY = 24 * HOUR;
WEEK = 7 * DAY;
MONTH = 4 * WEEK;
YEAR = 12 * MONTH;



t_1 = linspace(0,0.5,120);
t_2 = linspace(0,3,120);

score_ij = zeros(length(t_1), length(t_2));

for i = 1:length(t_1)
    for j = 1:length(t_2)
        score_ij(i, j) = SCORE_PATHS_GREEDY([t_1(i), 0, t_2(j)], LAUNCH_RANGE, ...
            LAUNCH_BODY, LANDING_BODY, GOAL_BODY, M_1, -1);

    end
end

[T_1, T_2] = meshgrid(t_1, t_2);

clf;

hold on;
grid minor

surf(T_1, T_2, score_ij')
view(3);
xlabel('t_1 (years, normalized decision vector)');
ylabel('t_2 (years, normalized decision vector)');
zlabel('Score (km/s, greedy leg total)');
title('Greedy leg score vs time variables');
grid on;

hold off;