%%
filepath = 'C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project';
T_choices = [3, 5, 7]; %All poissible choices of input size
N_choices = [100];
AR_1 = (0:0.05:0.95);
column_labels = {'AR_p','JLMS','NW','NW_conditional_w1w2', 'Local_Linear_Forest'}; %'Local_Linear_Forest'

%%
for i=1:length(N_choices)
    for j=1:length(T_choices)
        N_choice = N_choices(i);
        T_choice = T_choices(j);
        MSE_results = [];
        b = 1;
        for p=1:length(AR_1)
            rho = AR_1(p);
            simulation_results_files = dir([filepath filesep fullfile('panel_SFA_simulation_Results', 'Gaussian',sprintf('AR_1=%.2f', rho))]);
            for a=1:length(simulation_results_files)
                if strcmpi(simulation_results_files(a).name,'.') || strcmpi(simulation_results_files(a).name, '..')
                    continue
                else
                    if (contains(simulation_results_files(a).name, 'MSE_results')) && (contains(simulation_results_files(a).name, sprintf('N=%d', N_choice))) && (contains(simulation_results_files(a).name, sprintf('T=%d', T_choice)))
                        filename = simulation_results_files(a).name;
                        MSE_result = table2array(readtable([filepath filesep fullfile('panel_SFA_simulation_Results', 'Gaussian',sprintf('AR_1=%.2f', rho), sprintf('%s', filename))]));
                        MSE_results(b,:) = MSE_result;
                        b = b+1;
                    end
                end
            end
        end

        %% raw MSE plots
        MSE_results = array2table(cat(2,MSE_results), 'VariableNames',column_labels);
        MSE_results = sortrows(MSE_results, {'AR_p'}); %Sort data 
        
        if ~exist(convertCharsToStrings([filepath filesep fullfile('Figures','SFA_panel_simulation')]))
            mkdir(convertCharsToStrings([filepath filesep fullfile('Figures','SFA_panel_simulation')]));
        end
        
        %MSE Plots - all estimators
        figure()
        set(gcf,'position',[10,10,750,650])
        plot(MSE_results{:,1},MSE_results{:,2},'-', 'color','k','LineWidth',2) %JLMS
        hold on
        plot(MSE_results{:,1},MSE_results{:,3},'--', 'color','k','LineWidth',2) %NW
        hold on
        plot(MSE_results{:,1},MSE_results{:,4},':', 'color','k','LineWidth',2) %NW conditional W
        hold on 
        plot(MSE_results{:,1},MSE_results{:,5},'-*', 'color','k','LineWidth',2) %LLF
        
        title(sprintf('N = %d, T = %d',N_choice, T_choice), 'fontsize',16);
        xlabel('$\rho$', 'Interpreter','latex', 'fontsize',15);
        ylabel('$MSE(\tilde{u}_{it})$', 'Interpreter','latex',  'fontsize',14);
        xlim([0,0.95])
        legend('JLMS $E[u_{it}|\varepsilon_{it}]$', 'NW $E[u_{i}|\varepsilon_{it}]$', 'NW $E[u_{it}|\varepsilon_{i1}, \dots, \varepsilon_{iT}]$', 'LLF $E[u_{it}| \varepsilon_{i1}, \dots, \varepsilon_{iT}]$', 'fontsize',16,'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal', 'NumColumns',2);
        
        pause(5) %Allow dropbox folder permissions to sync with each export
        saveas(gcf,[filepath filesep fullfile('Figures','SFA_panel_simulation',sprintf('SFA_panel_simulation_all N=%d_T=%d.png', N_choice, T_choice))]);
        
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