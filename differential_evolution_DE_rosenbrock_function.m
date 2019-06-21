%% differential evolution
clear

f= @(x) Rosenbrock(x); % objective function
D = 2; % dimensions
R = [-5, 5]; % trust region
SetMinMax = -1; % min or max problem

%% paramters 

FE = 2500; % function evaluations
popSize = 20; % population size
F = 0.75; 

%% initialize 

% p-vectors 
x_cur = rand(popSize, D) * (R(2)-R(1)+R(1));                                             
f_cur = f(x_cur);

% find elitest
[~, ie] = max(f_cur*SetMinMax); % finde index of best solution (ie = index of elitest)
x_el = x_cur(ie,:);  
f_el = f_cur(ie,:);

% search 
for gen = 2:(FE/popSize)
    
    % create new offspring 
    m1 = randperm(popSize); % parent 1
    m2 = randperm(popSize); % parent 2
    m3 = randperm(popSize); % parent 3
    
    
    x_new = x_cur(m1,:) + F*(x_cur(m2,:)-x_cur(m3,:)); % create new population each x1, x2, x3 choosen randomly without replacement
    f_new= f(x_new); 

    % update population (by simple turnament) 
    update = f_new * SetMinMax > f_cur * SetMinMax;
    x_cur(update,:) = x_new(update,:); 
    f_cur(update,:) = f_new(update,:);
    
    % new elitest?
    [fe, ie] = max(f_cur*SetMinMax);
    if fe > f_el*SetMinMax 
          x_el=x_cur(ie,:);
          f_el=f_cur(ie);
    end   
end

%% report 


disp(f_el);