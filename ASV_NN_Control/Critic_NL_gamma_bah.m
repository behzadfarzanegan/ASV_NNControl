function output = Critic_NL_gamma_bah(x)


% syms x [1 6]
DS = [];

k=1;
while k <= length(x)/2
    for i=k:length(x)/2
      DS =[DS;x(k)*x(i)]; 
    end
    k=k+1;
end
for k=1:length(x)/2
    for j=k:length(x)/2
        for l=j:length(x)/2
            for i=l:length(x)/2
                DS =[DS;x(k)*x(j)*x(l)*x(i)];
            end
        end
    end
end

for k=1:length(x)/2
    for j=k:length(x)/2
        for l=j:length(x)/2
            for i=l:length(x)/2
                for ii= i:length(x)/2
                    for iii = ii:length(x)/2
                        DS =[DS;x(k)*x(j)*x(l)*x(i)*x(ii)*x(iii)];
                    end
                end
            end
        end
    end
end

while (k > length(x)/2 && k <= length(x))
    for i=k:length(x)
      DS =[DS;x(k)*x(i)]; 
    end
    k=k+1;
end


for k=length(x)/2+1:length(x)
    for j=k:length(x)
        for l=j:length(x)
            for i=l:length(x)
                DS =[DS;x(k)*x(j)*x(l)*x(i)];
            end
        end
    end
end
for k=length(x)/2+1:length(x)
    for j=k:length(x)
        for l=j:length(x)
            for i=l:length(x)
                for ii= i:length(x)
                    for iii = ii:length(x)
                        DS =[DS;x(k)*x(j)*x(l)*x(i)*x(ii)*x(iii)];
                    
                    end
                end
            end
        end
    end
end

output = DS;


end