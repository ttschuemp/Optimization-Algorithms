%% genetic algorithm (knapsack problem)
clear
% optimization problem 
value  = [ 12 10 9 5 23 4 11 17]; % assets 
weight = [1 5 3 4 15 20 13 7]; 

maxcapacity = 20; % constraint 
punishment = @(x) min(maxcapacity - x * weight', 0) * 20; 
f = @(x) x * value' + punishment(x); 


%% setting for method
FE = 1000; % function evaluations
popSize = 20; % population size 
maxGen = FE/popSize;
prob_mut = .05; % prob. mutation
prob_from_p2 = 0.5; % prob. for cross over

%% inital solution 
N = numel(value);  

x_cur = rand(popSize, N) < 0.5; % vector of binaries, each row is one individual
f_cur = f(x_cur); % fitness of each individual

[f_el, i_el] = max(f_cur); % find fitness and index of best solution 
x_el = x_cur(i_el,:); % binary rep of x_elitest / what assets are taken




for gen = 2:maxGen 
    % CROSS-OVER
    p2 = randperm(popSize); 
    x_new = x_cur; 
    XO = rand(popSize, N) < prob_from_p2;
    x_cur_perm = x_cur(p2,:);
    x_new(XO) = x_cur_perm(XO); % genes from parent 2
    
    % MUTATION
    i_mut = rand(popSize, N) < prob_mut; % only these few get flipped 
    x_new(i_mut) = ~x_new(i_mut); % flipp in opposit 
    f_new = f(x_new); % fitness/function evaluation of new candidate
    
    % FIND NEW POPULATION
    replace = f_new > f_cur; % logical vector 
    x_cur(replace, :) = x_new(replace,: ); 
    f_cur(replace) = f_new(replace); 
    
    % ELITEST 
    [fe, i_e] = max(f_cur);
    if fe > f_el
        f_el = fe; 
        x_el = x_cur(i_el,: ); 
    end 

     disp([gen, f_el, x_el]); 
     pause(.1);
end



