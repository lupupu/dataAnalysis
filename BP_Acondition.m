% �ô���Ϊ����BP�������Ԥ���㷨

%% ��ջ�������
clc
clear

%% ѵ������Ԥ��������ȡ����һ��
%���������������
load data input output

%��1��3341���������
k=rand(1,3341);
[m,n]=sort(k);

%�ҳ�ѵ�����ݺ�Ԥ������
input_train=input(n(1:3241),:)';
output_train=output(n(1:3241));
input_test=input(n(3242:3341),:)';
output_test=output(n(3242:3341));

%ѡ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% BP����ѵ��
% ��ʼ������ṹ
net=newff(inputn,outputn,7);

net.trainParam.epochs=100;%��������
net.trainParam.lr=0.00001;%ѧϰ��
net.trainParam.goal=0.02;%Ŀ��

%����ѵ��
net=train(net,inputn,outputn);

%% BP����Ԥ��
%Ԥ�����ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
 
%����Ԥ�����
an=sim(net,inputn_test);
 
%�����������һ��
BPoutput=mapminmax('reverse',an,outputps);

%% �������

figure(1)
plot(BPoutput,':og')
hold on
plot(output_test,'-*');
legend('Ԥ�����','�������')
title('BP����Ԥ�����','fontsize',12)
ylabel('�������','fontsize',12)
xlabel('����','fontsize',12)
%Ԥ�����
error=BPoutput-output_test;


figure(2)
plot(error,'-*')
title('BP����Ԥ�����','fontsize',12)
ylabel('���','fontsize',12)
xlabel('����','fontsize',12)

figure(3)
plot((output_test-BPoutput)./BPoutput,'-*');
title('������Ԥ�����ٷֱ�')

errorsum=sum(abs(error));

figure(4)
hist(output_test-BPoutput);

%% ���ܶȹ���ͼ
[f_ks1,xi1,u1] = ksdensity(output_test-BPoutput);
figure(5)
f1=plot(xi1,f_ks1,'b','linewidth',3);%���ƺ��ܶȹ���ͼ������������Ϊ��ɫʵ�ߣ��߿�Ϊ3