function struct2vars(s)
%STRUCT2VARS Extract values from struct fields to workspace variables
names = fieldnames(s);
for i = 1:numel(names)
    assignin('caller', names{i}, s.(names{i}));
end