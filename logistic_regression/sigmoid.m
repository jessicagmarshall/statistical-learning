function [ y ] = sigmoid( x )
%outputs the sigmoid function of the input

    y = 1 / (1 + exp(- x));

end

