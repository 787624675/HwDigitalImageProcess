本次作业是使用双线性插值来对图像进行放大和缩小（resize），同学们需要完成的是resize函数的书写。
编写完成之后，运行程序，如果书写正确，会打印出Pass；错误则不会。

这里提供两个学习（参考）的链接，避免大家踩坑。
<https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html>

<https://www.cnblogs.com/yssongest/p/5303151.html>

#### 额外一些事项

可以使用show_images来看一下gt和resized_img，具体用法如下：

```python
if __name__ == '__main__':
    ratios = [0.5, 0.8, 1.2, 1.5]

    img = cv2.imread('images/img_2.jpeg')   # type(img) = ndarray

    start_time = time.time()
    for ratio in ratios:
        gt = get_gt(img, ratio)
        resized_img = resize(img, ratio)
        
        if ratio == xxx:
        	show_images(gt, resized_img)   # 这里加就行

        judge(gt, resized_img, ratio)
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f'用时{total_time:.4f}秒')
    print('Pass')
```
#### 实验记录

- 缩小的时候第一个像素位置的选取会影响精度，应选取为
  - src_x = float((j + 0.5) * scale_w - 0.5)
  - src_y = float((i + 0.5) * scale_h - 0.5)
- 放大的时候，边界条件没有考虑导致精度不够
- 考虑了边界条件后虽然精度变高了，但是还是没有完全与opencv的结果相同
- 尝试了把opencv的c源码转换为python，在从源图像取像素的时候下标赋值错误，背影赋值成sx，sy类似的，却赋值成j和i
