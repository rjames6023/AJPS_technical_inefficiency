%%
filepath = 'C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project';
n_inputs_choices = (2:1:4); %All poissible choices of input size
sample_sizes = [300, 500];

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
        upper_quartile_MSE_results = [];
        mid_quartile_MSE_results = [];
        lower_quartile_MSE_results = [];
        d = 1;
        b = 1;
        c = 1;
        for a=1:length(simulation_results_files)
            if strcmpi(simulation_results_files(a).name,'.') || strcmpi(simulation_results_files(a).name, '..')
                continue
            else
                if (contains(simulation_results_files(a).name, 'upper_quartile_MSE_results')) && (contains(simulation_results_files(a).name, sprintf('n=%d', n)))
                    filename = simulation_results_files(a).name;
                    MSE_result = table2array(readtable([filepath filesep fullfile('SFA_simulation_Results', sprintf('n_inputs=%d', n_inputs), sprintf('%s', filename))]));
                    upper_quartile_MSE_results(b,:) = MSE_result;
                    b = b+1;
                elseif (contains(simulation_results_files(a).name, 'mid_quartile_MSE_results')) && (contains(simulation_results_files(a).name, sprintf('n=%d', n)))
                    filename = simulation_results_files(a).name;
                    MSE_result = table2array(readtable([filepath filesep fullfile('SFA_simulation_Results', sprintf('n_inputs=%d', n_inputs), sprintf('%s', filename))]));
                    mid_quartile_MSE_results(d,:) = MSE_result;
                    d = d +1;
                elseif (contains(simulation_results_files(a).name, 'lower_quartile_MSE_results')) && (contains(simulation_results_files(a).name, sprintf('n=%d', n)))
                    filename = simulation_results_files(a).name;
                    MSE_result = table2array(readtable([filepath filesep fullfile('SFA_simulation_Results', sprintf('n_inputs=%d', n_inputs), sprintf('%s', filename))]));
                    lower_quartile_MSE_results(c,:) = MSE_result;
                    c = c +1;
                end
            end
        end
        
        %% raw MSE plots
        upper_quartile_MSE_results = array2table(cat(2,upper_quartile_MSE_results), 'VariableNames',column_labels);
        upper_quartile_MSE_results = sortrows(upper_quartile_MSE_results, {column_labels{1:n_inputs-1}}); %Sort data 
   
        mid_quartile_MSE_results = array2table(cat(2,mid_quartile_MSE_results), 'VariableNames',column_labels);
        mid_quartile_MSE_results = sortrows(mid_quartile_MSE_results, {column_labels{1:n_inputs-1}}); %Sort data 
                
        lower_quartile_MSE_results = array2table(cat(2,lower_quartile_MSE_results), 'VariableNames',column_labels);
        lower_quartile_MSE_results = sortrows(lower_quartile_MSE_results, {column_labels{1:n_inputs-1}}); %Sort data 
        
        if ~exist(convertCharsToStrings([filepath filesep fullfile('Figures','SFA_simulation')]))
            mkdir(convertCharsToStrings([filepath filesep fullfile('Figures','SFA_simulation')]));
        end
        
        %Upper Quartile MSE Plots - all estimators
        figure()
        set(gcf,'position',[10,10,750,650])
        plot(upper_quartile_MSE_results{:,1},upper_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+1},'-','color','k','LineWidth',2) %JLMS %'color','k'
        hold on
        plot(upper_quartile_MSE_results{:,1},upper_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+2},'--','color','k','LineWidth',2) %NW %'color','b'
        hold on
        plot(upper_quartile_MSE_results{:,1},upper_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+3},':','color','k','LineWidth',2) %NW conditional W %'color','r'
        hold on
        plot(upper_quartile_MSE_results{:,1},upper_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+4},'-.','color','k','LineWidth',2) %NW conditional W

        title(sprintf('$1^{st}$ Quartile n = %d, J = %d',n, n_inputs), 'fontsize',16, 'Interpreter','latex');
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
        saveas(gcf,[filepath filesep fullfile('Figures','SFA_simulation',sprintf('upper_quartile_MSE_SFA_simulation_n_inputs=%d_n=%d.png', n_inputs, n))]);

        %Mid Quartile MSE Plots - all estimators
        figure()
        set(gcf,'position',[10,10,750,650])
        plot(mid_quartile_MSE_results{:,1},mid_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+1},'-','color','k','LineWidth',2) %JLMS %'color','k'
        hold on
        plot(mid_quartile_MSE_results{:,1},mid_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+2},'--','color','k','LineWidth',2) %NW %'color','b'
        hold on
        plot(mid_quartile_MSE_results{:,1},mid_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+3},':','color','k','LineWidth',2) %NW conditional W %'color','r'
        hold on
        plot(mid_quartile_MSE_results{:,1},mid_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+4},'-.','color','k','LineWidth',2) %NW conditional W
 
        title(sprintf('$2^{nd}$ and $3^{rd}$ Quartile n = %d, J = %d',n, n_inputs), 'fontsize',16, 'Interpreter','latex');
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
        saveas(gcf,[filepath filesep fullfile('Figures','SFA_simulation',sprintf('mid_quartile_MSE_SFA_simulation_n_inputs=%d_n=%d.png', n_inputs, n))]);

        %Lower Quartile MSE Plots - all estimators
        figure()
        set(gcf,'position',[10,10,750,650])
        plot(lower_quartile_MSE_results{:,1},lower_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+1},'-','color','k','LineWidth',2) %JLMS %'color','k'
        hold on
        plot(lower_quartile_MSE_results{:,1},lower_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+2},'--','color','k','LineWidth',2) %NW %'color','b'
        hold on
        plot(lower_quartile_MSE_results{:,1},lower_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+3},':','color','k','LineWidth',2) %NW conditional W %'color','r'
        hold on
        plot(lower_quartile_MSE_results{:,1},lower_quartile_MSE_results{:,n_inputs-1+n_rho_W_terms+4},'-.','color','k','LineWidth',2) %NW conditional W
 
        title(sprintf('$4^{th}$ Quartile n = %d, J = %d',n, n_inputs), 'fontsize',16, 'Interpreter','latex');
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
        saveas(gcf,[filepath filesep fullfile('Figures','SFA_simulation',sprintf('lower_quartile_MSE_SFA_simulation_n_inputs=%d_n=%d.png', n_inputs, n))]);
    end
end