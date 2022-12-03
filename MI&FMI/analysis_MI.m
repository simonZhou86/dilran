function MI = analysis_MI(A,B,F)

% MI_A = nmi(A,F);
% MI_B = nmi(B,F);

% MI_A = mutual_information_images(A,F);
% MI_B = mutual_information_images(B,F);

MI_A = MutualInformation(A,F);
MI_B = MutualInformation(B,F);

MI = (MI_A + MI_B) / 2;

end

% MutualInformation: returns mutual information (in bits) of the 'X' and 'Y'
% by Will Dwinnell
%
% I = MutualInformation(X,Y);
%
% I  = calculated mutual information (in bits)
% X  = variable(s) to be analyzed (column vector)
% Y  = variable to be analyzed (column vector)
%
% Note: Multiple variables may be handled jointly as columns in matrix 'X'.
% Note: Requires the 'Entropy' and 'JointEntropy' functions.
%
% Last modified: Nov-12-2006

function I = MutualInformation(X,Y)

if (size(X,2) > 1)  % More than one predictor?
    % Axiom of information theory
    I = JointEntropy(X) + entropy(Y) - JointEntropy([X Y]);
else
    % Axiom of information theory
    I = entropy(X) + entropy(Y) - JointEntropy([X Y]);
end


% God bless Claude Shannon.

% EOF
end


% JointEntropy: Returns joint entropy (in bits) of each column of 'X'
% by Will Dwinnell
%
% H = JointEntropy(X)
%
% H = calculated joint entropy (in bits)
% X = data to be analyzed
%
% Last modified: Aug-29-2006

function H = JointEntropy(X)

% Sort to get identical records together
X = sortrows(X);

% Find elemental differences from predecessors
DeltaRow = (X(2:end,:) ~= X(1:end-1,:));

% Summarize by record
Delta = [1; any(DeltaRow')'];

% Generate vector symbol indices
VectorX = cumsum(Delta);

% Calculate entropy the usual way on the vector symbols
H = entropy(VectorX);


% God bless Claude Shannon.

% EOF
end