function f = Rosenbrock(x,a,b)

if nargin==1
   a=1;
   b=100;
end

   f = (a-x(:,1)).^2 + b*(x(:,2)-x(:,1).^2).^2;
   
end