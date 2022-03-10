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
        upper_quartile_MSE_results = [];
        mid_quartile_MSE_results = [];
        lower_quartile_MSE_results = [];
        d = 1;
        b = 1;
        c = 1;
        for p=1:length(AR_1)
            rho = AR_1(p);
            simulation_results_files = dir([filepath filesep fullfile('panel_SFA_simulation_Results', 'Gaussian',sprintf('AR_1=%.2f', rho))]);
            for a=1:length(simulation_results_files)
                if strcmpi(simulation_results_files(a).name,'.') || strcmpi(simulation_results_files(a).name, '..')
                    continue
                else
                    if (contains(simulation_results_files(a).name, 'upper_quartile_MSE_results')) && (contains(simulation_results_files(a).name, sprintf('N=%d', N_choice))) && (contains(simulation_results_files(a).name, sprintf('T=%d', T_choice)))
                        filename = simulation_results_files(a).name;
                        MSE_result = table2array(readtable([filepath filesep fullfile('panel_SFA_simulation_Results', 'Gaussian',sprintf('AR_1=%.2f', rho), sprintf('%s', filename))]));
                        upper_quartile_MSE_results(b,:) = MSE_result;
                        b = b+1;
                    elseif (contains(simulation_results_files(a).name, 'mid_quartile_MSE_results')) && (contains(simulation_results_files(a).name, sprintf('N=%d', N_choice))) && (contains(simulation_results_files(a).name, sprintf('T=%d', T_choice)))
                        filename = simulation_results_files(a).name;
                        MSE_result = table2array(readtable([filepath filesep fullfile('panel_SFA_simulation_Results', 'Gaussian',sprintf('AR_1=%.2f', rho), sprintf('%s', filename))]));
                        mid_quartile_MSE_results(d,:) = MSE_result;
                        d = d+1;
                    elseif (contains(simulation_results_files(a).name, 'lower_quartile_MSE_results')) && (contains(simulation_results_files(a).name, sprintf('N=%d', N_choice))) && (contains(simulation_results_files(a).name, sprintf('T=%d', T_choice)))
                        filename = simulation_results_files(a).name;
                        MSE_result = table2array(readtable([filepath filesep fullfile('panel_SFA_simulation_Results', 'Gaussian',sprintf('AR_1=%.2f', rho), sprintf('%s', filename))]));
                        lower_quartile_MSE_results(c,:) = MSE_result;
                        c = c+1;
                    end
                end
            end
        end
        
        upper_quartile_MSE_results = array2table(cat(2,upper_quartile_MSE_results), 'VariableNames',column_labels);
        upper_quartile_MSE_results = sortrows(upper_quartile_MSE_results, {'AR_p'}); %Sort data
   
        mid_quartile_MSE_results = array2table(cat(2,mid_quartile_MSE_results), 'VariableNames',column_labels);
        mid_quartile_MSE_results = sortrows(mid_quartile_MSE_results, {'AR_p'}); %Sort data
                
        lower_quartile_MSE_results = array2table(cat(2,lower_quartile_MSE_results), 'VariableNames',column_labels);
        lower_quartile_MSE_results = sortrows(lower_quartile_MSE_results, {'AR_p'}); %Sort data 
        
        if ~exist(convertCharsToStrings([filepath filesep fullfile('Figures','SFA_panel_simulation')]))
            mkdir(convertCharsToStrings([filepath filesep fullfile('Figures','SFA_panel_simulation')]));
        end
        
        %MSE Plots - all estimators
        figure()
        set(gcf,'position',[10,10,750,650])
        plot(upper_quartile_MSE_results{:,1},upper_quartile_MSE_results{:,2},'-', 'color','k','LineWidth',2) %JLMS
        hold on
        plot(upper_quartile_MSE_results{:,1},upper_quartile_MSE_results{:,3},'--', 'color','k','LineWidth',2) %NW
        hold on
        plot(upper_quartile_MSE_results{:,1},upper_quartile_MSE_results{:,4},':', 'color','k','LineWidth',2) %NW conditional W
        hold on 
        plot(upper_quartile_MSE_results{:,1},upper_quartile_MSE_results{:,5},'-*', 'color','k','LineWidth',2) %LLF
        
        title(sprintf('Upper Quartile N = %d, T = %d',N_choice, T_choice), 'fontsize',16);
        xlabel('$\rho$', 'Interpreter','latex', 'fontsize',15);
        ylabel('$MSE(\tilde{u}_{it})$', 'Interpreter','latex',  'fontsize',14);
        xlim([0,0.95])
        legend('JLMS $E[u_{it}|\varepsilon_{it}]$', 'NW $E[u_{i}|\varepsilon_{it}]$', 'NW $E[u_{it}|\varepsilon_{i1}, \dots, \varepsilon_{iT}]$', 'LLF $E[u_{it}| \varepsilon_{i1}, \dots, \varepsilon_{iT}]$', 'fontsize',16,'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal', 'NumColumns',2);
        
        pause(5) %Allow dropbox folder permissions to sync with each export
        saveas(gcf,[filepath filesep fullfile('Figures','SFA_panel_simulation',sprintf('Upper Quartile SFA_panel_simulation_all N=%d_T=%d.png', N_choice, T_choice))]);
        
        figure()
        set(gcf,'position',[10,10,750,650])
        plot(mid_quartile_MSE_results{:,1},mid_quartile_MSE_results{:,2},'-', 'color','k','LineWidth',2) %JLMS
        hold on
        plot(mid_quartile_MSE_results{:,1},mid_quartile_MSE_results{:,3},'--', 'color','k','LineWidth',2) %NW
        hold on
        plot(mid_quartile_MSE_results{:,1},mid_quartile_MSE_results{:,4},':', 'color','k','LineWidth',2) %NW conditional W
        hold on 
        plot(mid_quartile_MSE_results{:,1},mid_quartile_MSE_results{:,5},'-*', 'color','k','LineWidth',2) %LLF

        title(sprintf('Mid Quartile N = %d, T = %d',N_choice, T_choice), 'fontsize',16);
        xlabel('$\rho$', 'Interpreter','latex', 'fontsize',15);
        ylabel('$MSE(\tilde{u}_{it})$', 'Interpreter','latex',  'fontsize',14);
        xlim([0,0.95])
        legend('JLMS $E[u_{it}|\varepsilon_{it}]$', 'NW $E[u_{i}|\varepsilon_{it}]$', 'NW $E[u_{it}|\varepsilon_{i1}, \dots, \varepsilon_{iT}]$', 'LLF $E[u_{it}| \varepsilon_{i1}, \dots, \varepsilon_{iT}]$', 'fontsize',16,'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal', 'NumColumns',2);
        
        pause(5) %Allow dropbox folder permissions to sync with each export
        saveas(gcf,[filepath filesep fullfile('Figures','SFA_panel_simulation',sprintf('Mid Quartile SFA_panel_simulation_all N=%d_T=%d.png', N_choice, T_choice))]);
        
        figure()
        set(gcf,'position',[10,10,750,650])
        plot(lower_quartile_MSE_results{:,1},lower_quartile_MSE_results{:,2},'-', 'color','k','LineWidth',2) %JLMS
        hold on
        plot(lower_quartile_MSE_results{:,1},lower_quartile_MSE_results{:,3},'--', 'color','k','LineWidth',2) %NW
        hold on
        plot(lower_quartile_MSE_results{:,1},lower_quartile_MSE_results{:,4},':', 'color','k','LineWidth',2) %NW conditional W
        hold on 
        plot(lower_quartile_MSE_results{:,1},lower_quartile_MSE_results{:,5},'-*', 'color','k','LineWidth',2) %LLF
        
        title(sprintf('Lower Quartile N = %d, T = %d',N_choice, T_choice), 'fontsize',16);
        xlabel('$\rho$', 'Interpreter','latex', 'fontsize',15);
        ylabel('$MSE(\tilde{u}_{it})$', 'Interpreter','latex',  'fontsize',14);
        xlim([0,0.95])
        legend('JLMS $E[u_{it}|\varepsilon_{it}]$', 'NW $E[u_{i}|\varepsilon_{it}]$', 'NW $E[u_{it}|\varepsilon_{i1}, \dots, \varepsilon_{iT}]$', 'LLF $E[u_{it}| \varepsilon_{i1}, \dots, \varepsilon_{iT}]$', 'fontsize',16,'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal', 'NumColumns',2);
        
        pause(5) %Allow dropbox folder permissions to sync with each export
        saveas(gcf,[filepath filesep fullfile('Figures','SFA_panel_simulation',sprintf('Lower Quartile SFA_panel_simulation_all N=%d_T=%d.png', N_choice, T_choice))]);
    end
end
