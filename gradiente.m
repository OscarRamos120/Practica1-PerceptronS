% Definir la nueva función f(x_1, x_2) = 10 - exp(-(x_1^2 + 3*x_2^2)).
syms x_1 x_2;
f = 10 - exp(-(x_1^2 + 3*x_2^2));

% Definir el rango en el que deseas calcular el gradiente.
x_1_range = -2:0.1:2;
x_2_range = -2:0.1:2;

% Inicializar matrices para almacenar el resultado del gradiente.
gradient_x_1 = zeros(length(x_1_range), length(x_2_range));
gradient_x_2 = zeros(length(x_1_range), length(x_2_range));

% Calcular el gradiente en el rango especificado.
for i = 1:length(x_1_range)
    for j = 1:length(x_2_range)
        gradient_x_1(i, j) = subs(diff(f, x_1), [x_1, x_2], [x_1_range(i), x_2_range(j)]);
        gradient_x_2(i, j) = subs(diff(f, x_2), [x_1, x_2], [x_1_range(i), x_2_range(j)]);
    end
end

% Crear una gráfica 3D del gradiente.
figure;
surf(x_1_range, x_2_range, gradient_x_1);
xlabel('X_1');
ylabel('X_2');
zlabel('Gradiente en X_1');
title('Gradiente en X_1 de la función f(x_1, x_2) = 10 - exp(-(x_1^2 + 3*x_2^2))');

figure;
surf(x_1_range, x_2_range, gradient_x_2);
xlabel('X_1');
ylabel('X_2');
zlabel('Gradiente en X_2');
title('Gradiente en X_2 de la función f(x_1, x_2) = 10 - exp(-(x_1^2 + 3*x_2^2))');
