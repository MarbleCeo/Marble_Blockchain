// Blockchain Dashboard Frontend Application

// Create the Vue application
const app = new Vue({
  el: '#app',
  data: {
    // Network status
    networkStatus: {
      onlineNodes: 0,
      totalNodes: 0,
      loading: true,
      error: null
    },
    // Blockchain statistics
    blockchainStats: {
      blocks: 0,
      transactions: 0,
      pendingTransactions: 0,
      avgBlockTime: 0,
      hashRate: 0,
      loading: true,
      error: null
    },
    // VMIA visualization data
    vmiaData: {
      nodes: [],
      links: [],
      loading: true,
      error: null
    },
    // Charts
    networkChart: null,
    blocksChart: null,
    vmiaChart: null,
    // Refresh intervals
    intervals: {
      network: null,
      blockchain: null,
      vmia: null
    }
  },
  
  computed: {
    // Network health percentage
    networkHealth() {
      if (this.networkStatus.totalNodes === 0) return 0;
      return Math.round((this.networkStatus.onlineNodes / this.networkStatus.totalNodes) * 100);
    },
    
    // Health status text and color
    healthStatus() {
      const health = this.networkHealth;
      if (health >= 90) {
        return { text: 'Excellent', color: '#10b981' };
      } else if (health >= 75) {
        return { text: 'Good', color: '#3b82f6' };
      } else if (health >= 50) {
        return { text: 'Fair', color: '#f59e0b' };
      } else {
        return { text: 'Poor', color: '#ef4444' };
      }
    }
  },
  
  methods: {
    // Fetch network status data
    fetchNetworkStatus() {
      this.networkStatus.loading = true;
      axios.get('/api/network/status')
        .then(response => {
          this.networkStatus.onlineNodes = response.data.online_nodes;
          this.networkStatus.totalNodes = response.data.total_nodes;
          this.updateNetworkChart();
          this.networkStatus.loading = false;
        })
        .catch(error => {
          console.error('Error fetching network status:', error);
          this.networkStatus.error = 'Failed to load network status data';
          this.networkStatus.loading = false;
        });
    },
    
    // Fetch blockchain statistics
    fetchBlockchainStats() {
      this.blockchainStats.loading = true;
      axios.get('/api/blockchain/stats')
        .then(response => {
          this.blockchainStats.blocks = response.data.total_blocks;
          this.blockchainStats.transactions = response.data.total_transactions;
          this.blockchainStats.pendingTransactions = response.data.pending_transactions;
          this.blockchainStats.avgBlockTime = response.data.avg_block_time;
          this.blockchainStats.hashRate = response.data.hash_rate;
          this.updateBlockchainCharts();
          this.blockchainStats.loading = false;
        })
        .catch(error => {
          console.error('Error fetching blockchain stats:', error);
          this.blockchainStats.error = 'Failed to load blockchain statistics';
          this.blockchainStats.loading = false;
        });
    },
    
    // Fetch VMIA neural network visualization data
    fetchVMIAVisualization() {
      this.vmiaData.loading = true;
      axios.get('/api/vmia/visualization')
        .then(response => {
          this.vmiaData.nodes = response.data.nodes;
          this.vmiaData.links = response.data.links;
          this.updateVMIAVisualization();
          this.vmiaData.loading = false;
        })
        .catch(error => {
          console.error('Error fetching VMIA visualization:', error);
          this.vmiaData.error = 'Failed to load VMIA visualization data';
          this.vmiaData.loading = false;
        });
    },
    
    // Initialize network status chart
    initNetworkChart() {
      const chartDom = document.getElementById('network-chart');
      if (!chartDom) return;
      
      this.networkChart = echarts.init(chartDom);
      this.updateNetworkChart();
    },
    
    // Update network status chart
    updateNetworkChart() {
      if (!this.networkChart) return;
      
      const option = {
        title: {
          text: 'Network Status',
          left: 'center'
        },
        tooltip: {
          trigger: 'item',
          formatter: '{a} <br/>{b}: {c} ({d}%)'
        },
        legend: {
          orient: 'vertical',
          left: 'left',
          data: ['Online Nodes', 'Offline Nodes']
        },
        series: [
          {
            name: 'Network Status',
            type: 'pie',
            radius: ['50%', '70%'],
            avoidLabelOverlap: false,
            itemStyle: {
              borderRadius: 10,
              borderColor: '#fff',
              borderWidth: 2
            },
            label: {
              show: false,
              position: 'center'
            },
            emphasis: {
              label: {
                show: true,
                fontSize: '20',
                fontWeight: 'bold'
              }
            },
            labelLine: {
              show: false
            },
            data: [
              { 
                value: this.networkStatus.onlineNodes, 
                name: 'Online Nodes',
                itemStyle: { color: '#10b981' } 
              },
              { 
                value: this.networkStatus.totalNodes - this.networkStatus.onlineNodes, 
                name: 'Offline Nodes',
                itemStyle: { color: '#ef4444' } 
              }
            ]
          }
        ]
      };
      
      this.networkChart.setOption(option);
    },
    
    // Initialize blockchain statistics charts
    initBlockchainCharts() {
      const chartDom = document.getElementById('blockchain-chart');
      if (!chartDom) return;
      
      this.blocksChart = echarts.init(chartDom);
      this.updateBlockchainCharts();
    },
    
    // Update blockchain statistics charts
    updateBlockchainCharts() {
      if (!this.blocksChart) return;
      
      const option = {
        title: {
          text: 'Blockchain Activity',
          left: 'center'
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow'
          }
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: [
          {
            type: 'category',
            data: ['Blocks', 'Transactions', 'Pending']
          }
        ],
        yAxis: [
          {
            type: 'value'
          }
        ],
        series: [
          {
            name: 'Count',
            type: 'bar',
            barWidth: '60%',
            data: [
              {
                value: this.blockchainStats.blocks,
                itemStyle: { color: '#3b82f6' }
              },
              {
                value: this.blockchainStats.transactions,
                itemStyle: { color: '#10b981' }
              },
              {
                value: this.blockchainStats.pendingTransactions,
                itemStyle: { color: '#f59e0b' }
              }
            ]
          }
        ]
      };
      
      this.blocksChart.setOption(option);
    },
    
    // Initialize VMIA neural network visualization
    initVMIAVisualization() {
      const chartDom = document.getElementById('vmia-visualization');
      if (!chartDom) return;
      
      this.vmiaChart = echarts.init(chartDom);
      this.updateVMIAVisualization();
    },
    
    // Update VMIA neural network visualization
    updateVMIAVisualization() {
      if (!this.vmiaChart) return;
      
      const categories = [
        { name: 'Input Layer' },
        { name: 'Hidden Layer 1' },
        { name: 'Hidden Layer 2' },
        { name: 'Output Layer' }
      ];
      
      // Transform the data to ECharts format
      const nodes = this.vmiaData.nodes.map(node => ({
        id: node.id,
        name: node.name || `Node ${node.id}`,
        symbolSize: node.value * 5 + 20,
        value: node.value,
        category: node.category,
        x: node.x,
        y: node.y,
        itemStyle: {
          color: node.active ? '#3b82f6' : '#9ca3af'
        }
      }));
      
      const links = this.vmiaData.links.map(link => ({
        source: link.source,
        target: link.target,
        value: link.value,
        lineStyle: {
          width: link.value * 2,
          color: link.active ? '#10b981' : '#d1d5db'
        }
      }));
      
      const option = {
        title: {
          text: 'VMIA Neural Network Visualization',
          left: 'center'
        },
        tooltip: {
          formatter: function(params) {
            if (params.dataType === 'node') {
              return `Node: ${params.data.name}<br>Value: ${params.data.value}`;
            } else {
              return `Connection: ${params.data.source} → ${params.data.target}<br>Weight: ${params.data.value}`;
            }
          }
        },
        legend: [{
          data: categories.map(a => a.name),
          orient: 'vertical',
          left: 10,
          top: 20
        }],
        animationDuration: 1500,
        animationEasingUpdate: 'quinticInOut',
        series: [
          {
            name: 'VMIA Neural Network',
            type: 'graph',
            layout: 'none',
            data: nodes,
            links: links,
            categories: categories,
            roam: true,
            label: {
              position: 'right',
              formatter: '{b}'
            },
            lineStyle: {
              color: 'source',
              curveness: 0.3
            },
            emphasis: {
              focus: 'adjacency',
              lineStyle: {
                width: 10
              }
            }
          }
        ]
      };
      
      this.vmiaChart.setOption(option);
    },
    
    // Helper method to resize charts when window size changes
    resizeCharts() {
      if (this.networkChart) this.networkChart.resize();
      if (this.blocksChart) this.blocksChart.resize();
      if (this.vmiaChart) this.vmiaChart.resize();
    },
    
    // Set up automatic refresh intervals
    setupRefreshIntervals() {
      // Clear any existing intervals
      this.clearRefreshIntervals();
      
      // Set new intervals
      this.intervals.network = setInterval(this.fetchNetworkStatus, 10000); // 10 seconds
      this.intervals.blockchain = setInterval(this.fetchBlockchainStats, 15000); // 15 seconds
      this.intervals.vmia = setInterval(this.fetchVMIAVisualization, 30000); // 30 seconds
    },
    
    // Clear all refresh intervals
    clearRefreshIntervals() {
      Object.values(this.intervals).forEach(interval => {
        if (interval) clearInterval(interval);
      });
    }
  },
  
  mounted() {
    // Initial data fetch
    this.fetchNetworkStatus();
    this.fetchBlockchainStats();
    this.fetchVMIAVisualization();
    
    // Initialize charts
    this.$nextTick(() => {
      this.initNetworkChart();
      this.initBlockchainCharts();
      this.initVMIAVisualization();
      
      // Set up automatic refresh
      this.setupRefreshIntervals();
      
      // Handle window resize
      window.addEventListener('resize', this.resizeCharts);
    });
  },
  
  beforeDestroy() {
    // Clean up intervals and event listeners
    this.clearRefreshIntervals();
    window.removeEventListener('resize', this.resizeCharts);
    
    // Dispose charts
    if (this.networkChart) this.networkChart.dispose();
    if (this.blocksChart) this.blocksChart.dispose();
    if (this.vmiaChart) this.vmiaChart.dispose();
  }
});

