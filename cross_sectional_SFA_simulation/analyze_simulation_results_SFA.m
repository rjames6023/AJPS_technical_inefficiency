%%
filepath = 'C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project';
n_inputs_choices = (2:1:4); %All poissible choices of input size
sample_sizes = [300, 500];
%%
for i=1:length(n_inputs_choices)
    for j=1:length(sample_sizes)
        n_inputs = n_inputs_choices(i);
        n = sample_sizes(j);
        if n_inputs == 2
            n_rho_W_terms = 0;
            column_labels = {'rho_u_w1','JLMS','NW','NW_conditional_w1w2', 'Local_Linear_Forest'};
        elseif n_inputs == 3
            n_rho_W_terms = 1;
            column_labels = {'rho_u_w1', 'rho_u_w2', 'rho_w1_w2', 'JLMS','NW','NW_conditional_w1w2', 'Local_Linear_Forest'};
        elseif n_inputs == 4
            n_rho_W_terms = 3;
            column_labels = {'rho_u_w1', 'rho_u_w2', 'rho_u_w3', 'rho_w1_w2', 'rho_w1_w3', 'rho_w2_w3', 'JLMS','NW','NW_conditional_w1w2','Local_Linear_Forest'};
        end
        simulation_results_files = dir([filepath filesep fullfile('SFA_simulation_Results', sprintf('n_inputs=%d', n_inputs))]);
        MSE_results = [];
        elementary_scores = [];
        b = 1;
        for a=1:length(simulation_results_files)
            if strcmpi(simulation_results_files(a).name,'.') || strcmpi(simulation_results_files(a).name, '..')
                continue
            else
                if (contains(simulation_results_files(a).name, 'u_hat_MSE_results')) && (contains(simulation_results_files(a).name, sprintf('n=%d', n)))
                    filename = simulation_results_files(a).name;
                    MSE_result = table2array(readtable([filepath filesep fullfile('SFA_simulation_Results', sprintf('n_inputs=%d', n_inputs), sprintf('%s', filename))]));
                    MSE_results(b,:) = MSE_result;
                    b = b+1;
                end
            end
        end

        %% raw MSE plots
        MSE_results = array2table(cat(2,MSE_results), 'VariableNames',column_labels);
        MSE_results = sortrows(MSE_results, {column_labels{1:n_inputs-1}}); %Sort data 
        
        if ~exist(convertCharsToStrings([filepath filesep fullfile('Figures','SFA_simulation')]))
            mkdir(convertCharsToStrings([filepath filesep fullfile('Figures','SFA_simulation')]));
        end
        
        %MSE Plots - all estimators
        figure()
        set(gcf,'position',[10,10,750,650])
        plot(MSE_results{:,1},MSE_results{:,n_inputs-1+n_rho_W_terms+1},'-','color','k','LineWidth',2) %JLMS %'color','k'
        hold on
        plot(MSE_results{:,1},MSE_results{:,n_inputs-1+n_rho_W_terms+2},'--','color','k','LineWidth',2) %NW %'color','b'
        hold on
        plot(MSE_results{:,1},MSE_results{:,n_inputs-1+n_rho_W_terms+3},':','color','k','LineWidth',2) %NW conditional W %'color','r'
        hold on
        plot(MSE_results{:,1},MSE_results{:,n_inputs-1+n_rho_W_terms+4},'-.','color','k','LineWidth',2) %NW conditional W
        ylim([0.1, 0.5])
        title(sprintf('n = %d, J = %d',n, n_inputs), 'fontsize',16);
        xlabel('$\rho_{u, \omega_{2}}$', 'fontsize',15, 'Interpreter','latex');
        ylabel('$MSE(\tilde{u}_{i})$', 'fontsize',14, 'Interpreter','latex');
        if n_inputs == 2
            legend('JLMS $E[u_{i}|\varepsilon_{i}]$', 'NW $E[u_{i}|\varepsilon_{i}]$', 'NW $E[u_{i}|\varepsilon_{i}, \omega_{i2}]$', 'LLF $E[u_{i}|\varepsilon_{i}, \omega_{i2}]$', 'fontsize',16 , 'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal', 'NumColumns',2);
        elseif n_inputs == 3
            legend('JLMS $E[u_{i}|\varepsilon_{i}]$', 'NW $E[u_{i}|\varepsilon_{i}]$', 'NW $E[u_{i}|\varepsilon_{i}, \omega_{i2}, \omega_{i3}]$', 'LLF $E[u_{i}|\varepsilon_{i}, \omega_{i2}, \omega_{i3}]$', 'fontsize',16, 'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal', 'NumColumns',2); 
        elseif n_inputs == 4
            legend('JLMS $E[u_{i}|\varepsilon_{i}]$', 'NW $E[u_{i}|\varepsilon_{i}]$', 'NW $E[u_{i}|\varepsilon_{i}, \omega_{i2}, \omega_{i3}, \omega_{i4}]$', 'LLF $E[u_{i}|\varepsilon_{i}, \omega_{i2}, \omega_{i3}, \omega_{i4}]$', 'fontsize',16, 'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal', 'NumColumns',2);
        end
        pause(2) %Allow dropbox folder permissions to sync with each export
        saveas(gcf,[filepath filesep fullfile('Figures','SFA_simulation',sprintf('SFA_simulation_n_inputs=%d_n=%d.png', n_inputs, n))]);
        
        %% Difference to benchmark plots
%         benchmark_estimator = 'JLMS';
%         other_estimators = MSE_results;
%         other_estimators(:,benchmark_estimator) = [];
%         [rows,cols] = size(other_estimators);
%         rep_benchmark = repmat(MSE_results{:,benchmark_estimator}, 1, cols-(n_inputs-1));
%         difference_to_benchmark_array = other_estimators{:, n_inputs:end} - rep_benchmark;
% 
%         %Difference of JLMS and non-parametric estimators
%         set(gcf,'position',[10,10,600,500])
%         plot(MSE_results{:,1},difference_to_benchmark_array(:,1),'b', MSE_results{:,1},difference_to_benchmark_array(:,2),'r',MSE_results{:,1},difference_to_benchmark_array(:,3),'g');
%         title(sprintf('n = %d',n));
%         xlabel('$\rho$', 'Interpreter','latex');
%         ylabel('$MSE(\hat{u})$', 'Interpreter','latex');
%         if n_inputs == 2
%             legend('NW $E[u|\epsilon]$', 'NW $E[u|\epsilon, \omega_{1}]$', 'LLF $E[u|\epsilon, \omega_{1}]$', 'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal');
%         elseif n_inputs == 3
%             legend('NW $E[u|\epsilon]$', 'NW $E[u|\epsilon, \omega_{1}, \omega_{2}]$', 'LLF $E[u|\epsilon, \omega_{1}, \omega_{2}]$', 'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal');
%         end
%         saveas(gcf,[filepath '\Figures' sprintf('\\non_SFA_simulation_difference_to_JLMS_n_inputs=%d_n=%d.png', n_inputs, n)]);

    end
end