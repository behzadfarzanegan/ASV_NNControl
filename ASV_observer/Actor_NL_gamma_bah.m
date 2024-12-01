function DS = Actor_NL_gamma_bah(X,v_actor,Neuron_Num_a)
    %---------------------------------------------------------------
    % Activity function for value function
    %---------------------------------------------------------------
    op =v_actor'*X;
    % with tanh activation funtion
    %DS =[tanh(op(1));tanh(op(2));tanh(op(3));tanh(op(4));tanh(op(5));tanh(op(6));tanh(op(7));tanh(op(8));tanh(op(9))];
    % with tanh activation function
    DS = [];
    for i=1:Neuron_Num_a
          DS =[DS;tanh(op(i))];
        % DS =[DS;logsig(op(i))];
    end
end