function theta = plot_Murphy_Diagram(scores, labels, colors, sigma2_u, filepath, n_inputs, n, export_label)
    %Find a range for theta
    max_tmp = sqrt(sigma2_u).*norminv((0.999+1)/2, 0, 1); %set the max theta to the 99.9th quantile of the distribution of technical inefficiency
    min_tmp = 0; %theoretical minimum of the technical inefficiency predictions
    tmp = [min_tmp-0.1*(max_tmp - min_tmp), max_tmp + 0.1*(max_tmp - min_tmp)];
    theta = linspace(tmp(1), tmp(2), 501);
    
    [~, cols] = size(scores);
    figure;
    hold on
    for i=1:cols
        plot(theta, scores(:,i), colors{i})
    end
    legend(labels, 'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal');
    xlabel('$\theta$', 'Interpreter','latex');
    ylabel('Expected Score', 'Interpreter','latex');
    saveas(gcf,[filepath filesep fullfile('Figures','Murphy Diagrams',sprintf('Murphy Diagram %s n_inputs=%d n=%d.png', export_label, n_inputs, n))]);
end