for dimNum = 1:3
    figure;
    for imNum = 1:64
        subplot(8,8,imNum), imshow(net16.params(1).value(:,:,dimNum,imNum),[])
    end
end